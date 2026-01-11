import os
import numpy as np
import rasterio
from rasterio.plot import show
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.ensemble import IsolationForest
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

import networkx as nx
import streamlit as st
import time
from datetime import datetime
import random
import gymnasium as gym
from gymnasium import spaces
import csv
import matplotlib.pyplot as plt

# Fix random seeds for determinism
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Constants
SENTINEL_DATA_DIR = "sentinel_data"
MODIS_DATA_DIR = "modis_data"
GRID_SIZE = 50
DRONE_COUNT = 3
LATENCY_THRESHOLD = 5.0
WIND_FACTOR = 0.1
TERRAIN_COST = 0.05
BATTERY_CAPACITY = 100.0
ENERGY_PER_MOVE = 1.0
IGNITION_POINTS = [(10, 10), (25, 25), (40, 20)]


# GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 768)  # Match embedding size
        self.resnet.to(device)

    def forward(self, x):
        return self.resnet(x)

class SatelliteIngestionAgent:
    def __init__(self):
        self.tiles = []
        self.ndvi_maps = []
        self.thermal_anomalies = []
        self.cached_embeddings = {}

    def load_tiles(self):
        try:
            sentinel_files = [f for f in os.listdir(SENTINEL_DATA_DIR) if f.endswith('.tif')]
            for file in sentinel_files[:3]:
                with rasterio.open(os.path.join(SENTINEL_DATA_DIR, file)) as src:
                    red = src.read(4)
                    nir = src.read(8)
                    blue = src.read(2)
                    green = src.read(3)
                    ndvi = (nir - red) / (nir + red + 1e-10)
                    self.ndvi_maps.append(ndvi)
                    rgb = np.stack([red, green, blue], axis=0)
                    self.tiles.append(rgb)
            modis_files = [f for f in os.listdir(MODIS_DATA_DIR) if f.endswith('.tif')]
            for file in modis_files[:1]:
                with rasterio.open(os.path.join(MODIS_DATA_DIR, file)) as src:
                    thermal = src.read(1)
                    anomaly = thermal > np.percentile(thermal, 95)
                    self.thermal_anomalies.append(anomaly)
        except FileNotFoundError:
            self.simulate_data()

    def simulate_data(self):
        for _ in range(3):
            tile = np.random.rand(3, GRID_SIZE, GRID_SIZE) * 255
            self.tiles.append(tile.astype(np.uint8))
            ndvi = np.random.rand(GRID_SIZE, GRID_SIZE) * 2 - 1
            self.ndvi_maps.append(ndvi)
        anomaly = np.random.rand(GRID_SIZE, GRID_SIZE) > 0.95
        self.thermal_anomalies.append(anomaly)

    def preprocess(self):
        processed_tiles = []
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        for tile in self.tiles:
            tile_resized = np.resize(tile, (3, 224, 224))
            tile_normalized = (tile_resized / 255.0 - mean) / std
            processed_tiles.append(tile_normalized)
        return processed_tiles

class ThreatDetectionAgent:
    def __init__(self):
        self.model = ResNetFeatureExtractor()
        self.model.eval()
        historical_features = []
        for _ in range(100):
            resnet_feat = np.random.rand(768)
            ndvi_feat = np.random.rand(1)
            thermal_feat = np.random.rand(1)
            historical_features.append(np.concatenate([resnet_feat, ndvi_feat, thermal_feat]))
        self.isolation_forest = IsolationForest(random_state=42)
        self.isolation_forest.fit(historical_features)

    def detect(self, tiles, ndvi_maps, thermal_anomalies, cache):
        detections = []
        for i, tile in enumerate(tiles):
            if i in cache:
                resnet_features = cache[i]
            else:
                inputs = torch.tensor(tile, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    resnet_features = self.model(inputs).cpu().numpy()
                cache[i] = resnet_features
            ndvi_vec = ndvi_maps[i].flatten().mean()
            thermal_vec = thermal_anomalies[min(i, len(thermal_anomalies)-1)].flatten().mean()  # Dynamic fusion
            fused_features = np.concatenate([resnet_features.flatten(), [ndvi_vec, thermal_vec]])
            anomaly_score = self.isolation_forest.predict([fused_features])[0]
            detections.append(anomaly_score == -1)
        return detections

class DecisionAgent:
    def __init__(self):
        pass

    def score_risks(self, detections):
        risks = [1.0 if det else 0.0 for det in detections]
        return risks

    def calibrate_confidence(self, risks):
        confidences = [min(1.0, r + np.random.normal(0, 0.1)) for r in risks]
        return confidences

    def escalate(self, risks, confidences):
        actions = ["deploy_drones" if r > 0.5 and c > 0.8 else "monitor" for r, c in zip(risks, confidences)]
        return actions

class DroneSwarmAgent:
    def __init__(self, drone_count):
        self.drone_count = drone_count
        self.positions = [(GRID_SIZE//2 + i*10, GRID_SIZE//2 + i*10) for i in range(drone_count)]
        self.batteries = [BATTERY_CAPACITY] * drone_count
        self.roles = ["scout", "containment", "monitor"] * (drone_count // 3 + 1)
        self.roles = self.roles[:drone_count]

    def coordinate(self, actions, twin):
        if "deploy_drones" in actions:
            for i in range(self.drone_count):
                role = self.roles[i]
                if role == "scout":
                    target = self.find_unexplored_target(twin)
                elif role == "containment":
                    target = self.find_fire_edge(twin)
                else:  # monitor
                    target = self.positions[i]  # Stay put or random
                if target:
                    path = self.a_star_path(self.positions[i], target)
                    if path and len(path) > 1:
                        self.positions[i] = path[1]
                        self.batteries[i] = max(0, self.batteries[i] - ENERGY_PER_MOVE)  # Clamp battery

    def find_unexplored_target(self, twin):
        # Simple: random unexplored node
        unexplored = [n for n in twin.graph.nodes if not twin.graph.nodes[n].get('visited', False)]
        return random.choice(unexplored) if unexplored else None

    def find_fire_edge(self, twin):
        # Target node adjacent to fire
        fire_nodes = [n for n in twin.graph.nodes if twin.graph.nodes[n]['on_fire']]
        edges = set()
        for node in fire_nodes:
            edges.update(twin.graph.neighbors(node))
        non_fire_edges = [e for e in edges if not twin.graph.nodes[e]['on_fire']]
        return random.choice(non_fire_edges) if non_fire_edges else None

    def a_star_path(self, start, goal):
        path = [start]
        while path[-1] != goal and len(path) < 50:
            current = path[-1]
            neighbors = [(current[0]+dx, current[1]+dy) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]]
            neighbors = [n for n in neighbors if 0 <= n[0] < GRID_SIZE and 0 <= n[1] < GRID_SIZE]
            if not neighbors:
                break
            path.append(min(neighbors, key=lambda n: abs(n[0]-goal[0]) + abs(n[1]-goal[1])))
        return path

class ContainmentAgent:
    def __init__(self):
        pass

    def plan_firebreaks(self, graph, positions, wind_dir):
        for pos in positions:
            neighbors = list(graph.neighbors(pos))
            for n in neighbors:
                if graph.has_edge(pos, n):
                    dx, dy = n[0] - pos[0], n[1] - pos[1]
                    if dx == wind_dir[0] and dy == wind_dir[1]:
                        graph.remove_edge(pos, n)

    def target_retardant(self, graph, positions):
        for pos in positions:
            if pos in graph.nodes:
                fuel_load = graph.nodes[pos].get('fuel', 1.0)
                graph.nodes[pos]['spread_prob'] *= np.exp(-0.5 / fuel_load)  # Nonlinear effect

class ForestDigitalTwin:
    def __init__(self):
        self.graph = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)
        self.wind_dir = (1, 0)  # Initial wind
        for node in self.graph.nodes:
            self.graph.nodes[node]['spread_prob'] = 0.01
            self.graph.nodes[node]['on_fire'] = node in IGNITION_POINTS
            self.graph.nodes[node]['fuel'] = np.random.rand() + 0.5
            self.graph.nodes[node]['visited'] = False

    def update_wind(self):
        # Per-step wind change
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        self.wind_dir = random.choice(directions)

    def update_fire_spread(self):
        self.update_wind()
        for node in list(self.graph.nodes):
            if self.graph.nodes[node]['on_fire']:
                neighbors = list(self.graph.neighbors(node))
                for n in neighbors:
                    if not self.graph.nodes[n]['on_fire']:
                        prob = self.graph.nodes[n]['spread_prob'] + WIND_FACTOR + TERRAIN_COST
                        dx, dy = n[0] - node[0], n[1] - node[1]
                        if dx == self.wind_dir[0] and dy == self.wind_dir[1]:
                            prob += 0.1
                        if np.random.rand() < prob:
                            self.graph.nodes[n]['on_fire'] = True

    def update_from_drones(self, positions):
        for pos in positions:
            self.graph.nodes[pos]['on_fire'] = False
            self.graph.nodes[pos]['visited'] = True

    def get_observation(self, ndvi_map):
        fire_map = np.zeros((GRID_SIZE, GRID_SIZE))
        for node in self.graph.nodes:
            if self.graph.nodes[node]['on_fire']:
                fire_map[node] = 1
        # Normalize NDVI to [0,1] and fire to [0,1]
        ndvi_normalized = (ndvi_map + 1) / 2  # From [-1,1] to [0,1]
        obs = np.stack([ndvi_normalized, fire_map], axis=0)  # Multi-channel: NDVI, fire
        return obs.astype(np.float32)

class MultiDroneEnv(gym.Env):
    def __init__(self, twin, swarm, ndvi_map):
        super(MultiDroneEnv, self).__init__()
        self.twin = twin
        self.swarm = swarm
        self.ndvi_map = ndvi_map
        self.action_space = spaces.MultiDiscrete([5] * self.swarm.drone_count)  # 0: up, 1: down, 2: left, 3: right, 4: stay
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, GRID_SIZE, GRID_SIZE), dtype=np.float32)  # Normalized [0,1]
        self.current_obs = self.twin.get_observation(self.ndvi_map)

    def step(self, actions):
        rewards = []
        for i, action in enumerate(actions):
            pos = self.swarm.positions[i]
            if action == 0: pos = (max(0, pos[0]-1), pos[1])  # Up
            elif action == 1: pos = (min(GRID_SIZE-1, pos[0]+1), pos[1])  # Down
            elif action == 2: pos = (pos[0], max(0, pos[1]-1))  # Left
            elif action == 3: pos = (pos[0], min(GRID_SIZE-1, pos[1]+1))  # Right
            # elif action == 4: stay
            self.swarm.positions[i] = pos
            self.swarm.batteries[i] = max(0, self.swarm.batteries[i] - ENERGY_PER_MOVE if action != 4 else self.swarm.batteries[i])  # Clamp
            reward = -ENERGY_PER_MOVE if action != 4 else 0
            if self.twin.graph.nodes[pos]['on_fire']:
                reward += 10
            rewards.append(reward)
        self.twin.update_from_drones(self.swarm.positions)
        self.twin.update_fire_spread()
        fire_count = sum(1 for n in self.twin.graph.nodes if self.twin.graph.nodes[n]['on_fire'])
        coverage_bonus = 0.1 * len(set(self.swarm.positions))
        containment_bonus = 10 * sum(1 for pos in self.swarm.positions if self.twin.graph.nodes[pos]['on_fire'])
        total_reward = sum(rewards) - fire_count * 0.1 + coverage_bonus + containment_bonus
        done = all(b <= 0 for b in self.swarm.batteries) or (fire_count == 0 and np.random.rand() > 0.05)  # Slight reignition chance
        self.current_obs = self.twin.get_observation(self.ndvi_map)
        terminated = done
        truncated = False
        info = {}

        return self.current_obs, total_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_obs = self.twin.get_observation(self.ndvi_map)
        info = {}
        return self.current_obs, info
    
    
def compute_metrics(twin, initial_fire_count, energy_consumed, start_time):
    current_fire_count = sum(1 for n in twin.graph.nodes if twin.graph.nodes[n]['on_fire'])
    area_saved = initial_fire_count - current_fire_count
    time_to_containment = time.time() - start_time
    co2_avoided = area_saved * 100
    return area_saved, time_to_containment, energy_consumed, co2_avoided

def log_metrics_to_csv(metrics, filename="metrics.csv"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, 'a', newline='') as csvfile:  # Append mode
        writer = csv.writer(csvfile)
        if os.stat(filename).st_size == 0:  # Write header if file is empty
            writer.writerow(["Timestamp", "Area Saved", "Time to Containment", "Energy Consumed", "CO2 Avoided"])
        writer.writerow([timestamp] + list(metrics))

def main():
    st.title("Autonomous Deforestation & Wildfire Detection System")

    # === SYSTEM INITIALIZATION ===
    ingestion = SatelliteIngestionAgent()
    detection = ThreatDetectionAgent()
    decision = DecisionAgent()
    swarm = DroneSwarmAgent(DRONE_COUNT)
    containment = ContainmentAgent()
    twin = ForestDigitalTwin()

    start_time = time.time()

    # === DATA INGESTION ===
    ingestion.load_tiles()
    tiles = ingestion.preprocess()
    ndvi_maps = ingestion.ndvi_maps
    thermal_anomalies = ingestion.thermal_anomalies

    # === THREAT DETECTION PIPELINE ===
    detections = detection.detect(
        tiles, ndvi_maps, thermal_anomalies, ingestion.cached_embeddings
    )
    risks = decision.score_risks(detections)
    confidences = decision.calibrate_confidence(risks)
    actions = decision.escalate(risks, confidences)

    swarm.coordinate(actions, twin)

    # === RL ENVIRONMENTS ===

    # === BASE ENV (REAL ENV FOR VISUALIZATION) ===
    base_env = MultiDroneEnv(twin, swarm, ndvi_maps[0])
    
    # === RL ENVIRONMENTS ===
    train_env = DummyVecEnv([lambda: base_env])
    
    eval_env = DummyVecEnv([
        lambda: MultiDroneEnv(
            ForestDigitalTwin(),
            DroneSwarmAgent(DRONE_COUNT),
            ndvi_maps[0]
        )
    ])
    
    # === CALLBACKS (CORRECT USAGE) ===
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=50,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=1000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1
    )

    # === PPO MODEL ===
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./ppo_fire_tensorboard/"
    )

    # === TRAINING ===


    if st.button("🚀 Start Simulation"):
        progress = st.progress(0)
        status = st.empty()
    
        with st.spinner("Training PPO agent..."):
            for i in range(5):
                model.learn(total_timesteps=400, reset_num_timesteps=False)
                progress.progress((i + 1) * 20)
                status.text(f"Training progress: {(i + 1) * 20}%")
    
        st.success("✅ Training complete!")
    


    # === POST-RL CONTAINMENT LOGIC ===
    containment.plan_firebreaks(twin.graph, swarm.positions, twin.wind_dir)
    containment.target_retardant(twin.graph, swarm.positions)
    twin.update_fire_spread()

    # === METRICS ===
    initial_fire_count = len(IGNITION_POINTS)
    energy_consumed = sum(BATTERY_CAPACITY - b for b in swarm.batteries)

    area_saved, time_reduction, energy, co2 = compute_metrics(
        twin,
        initial_fire_count,
        energy_consumed,
        start_time
    )

    latency = time.time() - start_time
    if latency > LATENCY_THRESHOLD:
        st.warning("Latency exceeded threshold. Switching to monitor-only mode.")

    # === STREAMLIT OUTPUT ===
    st.write(f"Latency: {latency:.2f}s")
    st.write(f"Area Saved: {area_saved}")
    st.write(f"Time to Containment: {time_reduction:.2f}s")
    st.write(f"Energy Consumed: {energy}")
    st.write(f"CO2 Avoided: {co2}")



    log_metrics_to_csv([area_saved, time_reduction, energy, co2])
    # === FULL VISUALIZATION (UPGRADED & SAFE) ===
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # NDVI layer (from real env, not DummyVecEnv)
    ndvi_layer = base_env.current_obs[0]
    ax.imshow(ndvi_layer, cmap="viridis")
    
    ax.set_title("NDVI Map with Drone Swarm & Active Fires")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Plot drone positions
    for i, pos in enumerate(swarm.positions):
        ax.scatter(
            pos[1], pos[0],
            c="red",
            s=60,
            edgecolors="black",
            label="Drone" if i == 0 else ""
        )
    
    # Plot fire locations
    fire_positions = [
        n for n in twin.graph.nodes
        if twin.graph.nodes[n].get("on_fire", False)
    ]
    
    for i, pos in enumerate(fire_positions):
        ax.scatter(
            pos[1], pos[0],
            c="orange",
            s=25,
            alpha=0.6,
            label="Fire" if i == 0 else ""
        )
    
    ax.legend(loc="upper right")
    ax.grid(False)
    
    st.pyplot(fig)
    
    # === METRIC LOGGING ===
    log_metrics_to_csv([
        area_saved,
        time_reduction,
        energy,
        co2
    ])
    
if __name__ == "__main__":
    main()
