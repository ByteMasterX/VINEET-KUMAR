

---

# **README / Overview of the Autonomous Deforestation & Wildfire Detection System**

## **Project Summary**

This project simulates an **autonomous wildfire detection and containment system** using drones, satellite imagery, and AI.
It uses reinforcement learning (PPO) to coordinate a **swarm of drones** to prevent the spread of forest fires.

---

## **System Components**

### 1️⃣ **SatelliteIngestionAgent**

* **Purpose:** Loads satellite imagery (Sentinel/MODIS) to detect vegetation health and thermal anomalies.
* **Key Steps:**

  * Loads `.tif` files for **NDVI (Normalized Difference Vegetation Index)** and thermal data.
  * If real data not available → **simulates data randomly**.
  * Preprocesses images for the model (resize + normalize).

**Key terms:**

* **NDVI:** Measures vegetation health. Range [-1, 1].
* **Thermal anomalies:** Areas that are unusually hot → could indicate fire.

---

### 2️⃣ **ThreatDetectionAgent**

* Uses a **ResNet18 model** for feature extraction from satellite images.
* Combines features with NDVI and thermal data.
* Uses **Isolation Forest** to detect anomalies (fire-prone areas).

**Key terms:**

* **ResNet18:** Neural network for feature extraction from images.
* **Isolation Forest:** ML model to detect anomalies (outliers).

---

### 3️⃣ **DecisionAgent**

* Scores risk based on threat detection.
* Computes confidence for each risk.
* Decides **action**:

  * **deploy_drones** → if high risk & high confidence
  * **monitor** → if low risk

**Key terms:**

* **Confidence:** Likelihood that the detected threat is real.
* **Action escalation:** Deciding whether drones should intervene.

---

### 4️⃣ **DroneSwarmAgent**

* Controls multiple drones.
* Each drone has a **role**: scout, containment, monitor.
* Drones move using a simple **A* pathfinding** logic to reach targets.

**Key terms:**

* **Scout:** Explore unexplored areas.
* **Containment:** Fight fire edges.
* **Monitor:** Stay put or observe.
* **Battery:** Energy consumed per move.

---

### 5️⃣ **ContainmentAgent**

* Plans **firebreaks** to prevent fire spread.
* Applies **retardant** to reduce fuel load and slow down fire.

**Key terms:**

* **Firebreaks:** Paths cleared to stop fire.
* **Fuel load:** Amount of combustible material.

---

### 6️⃣ **ForestDigitalTwin**

* Digital simulation of the forest as a **grid graph**.
* Nodes = forest patches, edges = adjacency.
* Attributes per node:

  * `on_fire` → is this patch burning
  * `fuel` → amount of flammable material
  * `spread_prob` → probability fire spreads to this patch
  * `visited` → if drone has visited

**Key terms:**

* **Digital twin:** Virtual model of a real-world system.
* **Wind factor & terrain cost:** Affect fire spread probability.

---

### 7️⃣ **MultiDroneEnv (Gymnasium Environment)**

* Custom **RL environment** for training drones.
* **Action space:** MultiDiscrete (up/down/left/right/stay) per drone.
* **Observation space:** NDVI + fire map (2 channels).
* **Reward:** Encourages putting out fire and minimizing energy.

**Key terms:**

* **Reward:** Positive points for extinguishing fire, negative for energy waste.
* **Episode done:** Drones depleted or fire fully contained.

---

## **Simulation Flow**

1. **System Initialization**

   * All agents (Satellite, Threat, Decision, Drone, Containment) are initialized.
   * Digital twin created for forest.

2. **Data Ingestion**

   * Load satellite tiles and NDVI maps.
   * Simulate if data is missing.

3. **Threat Detection**

   * Detect potential fire spots via ML model.
   * Compute risk and confidence.
   * Decide actions (deploy drones or monitor).

4. **Drone Movement**

   * Drones move according to roles.
   * Fire updates in digital twin based on spread rules.

5. **Reinforcement Learning**

   * PPO trains drones for **optimal containment policy**.
   * DummyVecEnv wraps the environment for Stable-Baselines3.
   * Stop training when reward threshold reached (example: 50).

6. **Containment & Metrics**

   * Plan firebreaks & apply retardant.
   * Update fire spread post-drone intervention.
   * Compute metrics.

---

## **Metrics Explained**

| Metric                  | Meaning                                                        |
| ----------------------- | -------------------------------------------------------------- |
| **Latency**             | Time system took to run the simulation. Low = faster response. |
| **Area Saved**          | How many initial fire patches were **successfully contained**. |
| **Time to Containment** | Time elapsed until fire is mostly contained.                   |
| **Energy Consumed**     | Sum of drone battery usage during simulation.                  |
| **CO2 Avoided**         | Environmental impact: more fire contained → more CO₂ saved.    |

**Note:** If `Area Saved = 0` and `CO2 Avoided = 0`, it means drones didn’t successfully contain any fires in this run.

---

## **Visualization**

* **NDVI Map** → background of forest health.
* **Drones** → red dots.
* **Active fires** → orange dots.
* Updates dynamically each step in simulation.

---

## **How to Use**

1. Install dependencies: `pip install torch torchvision rasterio stable-baselines3 gymnasium matplotlib networkx streamlit`
2. Run Streamlit app:

```bash
streamlit run wildfire_sim.py
```

3. Press **"Start Simulation"** → Watch drones react, fires spread, and metrics update.
4. Metrics logged in `metrics.csv` for analysis.

---

## **Key Features**

* Integrates **satellite imagery**, **ML anomaly detection**, **digital twin modeling**, and **drone swarm control**.
* Real-time **metric reporting**: area saved, CO₂ avoided, energy used.
* **RL-based drone coordination** using PPO.

---

