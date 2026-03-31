# 🌆 AI-Driven Urban Computing & Planning System

## 📌 Overview

This project presents a **comprehensive AI-powered urban computing framework** designed to analyze, forecast, simulate, and optimize city-scale infrastructure systems.

Built from scratch, the system integrates:

* Data engineering
* Statistical modeling
* Machine learning & deep learning
* Network science (cascading failures)
* Optimization techniques

The framework is applied to **Bengaluru, India**, addressing real-world challenges such as:

* Traffic congestion 🚦
* Infrastructure stress ⚡
* Urban expansion 🏙️
* System resilience 🔗

---

## 🚀 Project Pipeline

### 🔹 1. Data Collection & Preparation

* Aggregated multi-source urban datasets:

  * Road networks (OSM)
  * Building footprints
  * Traffic data
  * BESCOM,BDA,KML
  * Opencity
* Performed:

  * Data cleaning
  * Feature engineering
  * Spatial transformations

---

### 🔹 2. Statistical Modeling & Inference

Used classical statistical techniques for **descriptive analysis and causal understanding**:

* **OLS Regression** → relationship modeling
* **Poisson Models** → count-based urban events
* **Clustering:**

  * K-Means
  * DBSCAN
* **Causal Inference:**

  * Directed Acyclic Graphs (DAGs)
  * Robustness validation

📊 Output:

* Insights into urban structure
* Identification of key influencing variables

---

### 🔹 3. AI-Based Forecasting (Multi-Horizon)

A hierarchical forecasting system across time scales:

#### 📅 Short-term (1 Year)

* SARIMAX
* ETS
* STL decomposition

#### 📅 Medium-term (5 Years)

* LightGBM
* Random Forest

#### 📅 Long-term (10 Years)

* LSTM
* N-BEATS
* Temporal Fusion Transformer (TFT)

📈 Predicts:

* Traffic congestion trends
* Urban growth patterns

---

### 🔹 4. Cascading Failure Modeling

Simulated infrastructure interdependencies using **network science**:

* Dependency Graph Construction
* Propagation Matrix Modeling
* Failure Detection Mechanism
* Cascade Simulation

🔍 Result:

* No significant failure chains detected
* Indicates system resilience under modeled conditions

---

### 🔹 5. Urban Redesign Engine (Novel Contribution)

Proposed a **custom multi-sector redesign framework**:

* Divided city into **5 functional sectors**
* Applied AI-driven restructuring for:

  * Traffic redistribution
  * Spatial optimization
  * Infrastructure balancing

🔗 **Redesign Output:**
👉 file:///C:/AIurban-planning/reports/redesign_map.html

---

### 🔹 6. Optimization (MILP-Based)

Optimized redesign layouts using:

* **Mixed Integer Linear Programming (MILP)**

🎯 Objectives:

* Minimize congestion
* Improve accessibility
* Balance infrastructure load

---

### 🔹 7. Geospatial Visualization (Folium)

Interactive maps built using **Folium**:

* Traffic congestion hotspots
* Building density distribution
* Spatial clusters

🔗 **Interactive Map:**
👉 *file:///C:/AIurban-planning/reports/maps/folium_failmap_20251206T133955Z.html*

---

### 🔹 8. Dashboard

Interactive dashboard for real-time exploration of results:

* Forecast outputs
* Statistical summaries
* Optimization insights

🔗 **Dashboard:**
👉 https://drive.google.com/drive/folders/1nBwCJwg93HqFtUaid3SSZrT2ugK0QDAo

📸 **Dashboard Preview:**

![Dashboard Screenshot](https://github.com/user-attachments/assets/cd773ae7-3bee-404f-becc-9a84a2d0881a)

---

## 📂 Project Structure

```text
AIurban-planning/
│
├── src/                      # Core application
├── AI_forecasting/           # Forecasting models
├── statistical_inference/    # OLS, Poisson, DAG analysis
├── cascade_model/            # Failure simulation
├── optimization/             # MILP optimization
├── redesign/                 # Urban redesign logic
├── dashboard/                # Dashboard app
├── notebooks/                # Experiments
├── data/                     # (excluded - large files)
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/saimadhuri2008/AIUrban-Computing.git
cd AIUrban-Computing
pip install -r requirements.txt
```

---

## ▶️ Running the Dashboard

```bash
streamlit run src/app.py
```

---



## 🧠 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* LightGBM
* Deep Learning (LSTM, TFT, N-BEATS)
* Statsmodels
* Folium
* Optimization (MILP)

---

## 📈 Key Contributions

* End-to-end urban AI pipeline
* Multi-horizon forecasting framework
* Cascading failure simulation
* Custom 5-sector redesign model
* Optimization-driven planning

---

## 🚀 Future Work

* Real-time streaming data integration
* Smart city IoT integration
* Reinforcement learning for adaptive planning
* Deployment at large-scale urban systems

---

## 👤 Author

**J.Sai Madhuri**
GitHub: https://github.com/saimadhuri2008

---

## 📜 License

MIT License
