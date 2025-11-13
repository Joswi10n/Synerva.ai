#  Synerva Cleanroom-AI  
**AI-Driven IoT Platform for Cleanroom Monitoring and Energy Optimization**

Synerva Cleanroom-AI is a full-stack system for **real-time cleanroom environmental monitoring, contamination prediction, and HVAC energy optimization**.  
It integrates **industrial-grade IoT sensors**, **Kafka/InfluxDB data pipelines**, and **deep-learning models** to provide actionable insights for energy-efficient contamination control.

---

##  Project Overview

Modern semiconductor and precision-manufacturing cleanrooms consume enormous energy to maintain air purity and temperature stability.  
Synerva Cleanroom-AI aims to **predict contamination events** and **reduce HVAC energy usage** without compromising ISO 14644 standards.

### Objectives
- Continuous ingestion of multi-parameter cleanroom data (PM 1/2.5/10, CO₂, temperature, humidity, pressure).  
- Real-time data streaming via **MQTT / Kafka → InfluxDB 2.0**.  
- Visualization through **Grafana dashboards** for occupancy, particle density, and environmental health.  
- Training LSTM and regression models for contamination prediction and adaptive HVAC control.  

---

##  System Architecture

```text
[ NCD Sensors ] 
      │
      ▼
[ MQTT / Kafka Stream ]
      │
      ▼
[ InfluxDB 2.0 Time-Series Store ]
      │
      ▼
[ Python Analytics + AI Models ]
      │
      ▼
[ Grafana Dashboard & BMS Insights ]


Core Components
Layer	Technology	Role
Sensors	NCD.io PM/CO₂/T/H/P modules	Data acquisition
Stream Transport	MQTT, Kafka	Real-time message bus
Database	InfluxDB 2.0	Time-series storage
Analytics / AI	Python (NumPy, Pandas, TensorFlow, PyTorch)	LSTM + Linear models
Visualization	Grafana	Live metrics dashboard
Deployment	Docker / Compose / ECS	Cloud execution
 Setup & Installation
Prerequisites

Python ≥ 3.10

InfluxDB 2.0 server (local or cloud)

MQTT broker (Mosquitto / EMQX)

Optional: Docker & Docker Compose

 Clone the Repository
git clone https://github.com/Joswi10n/Synerva.ai.git
cd Synerva.ai

 Create Environment
python -m venv venv
venv\Scripts\activate  # (Windows)
# or
source venv/bin/activate  # (Linux/Mac)
pip install -r requirements.txt

 Configure Environment

Copy and update the example configuration:

cp config/config.example.env .env

 Run Data Ingestion / Prediction
python input_to_influx.py          # Stream sensor data into InfluxDB
python predict_once.py --help      # Run one-time prediction

 Live Dashboard

Once data is ingested into InfluxDB, connect Grafana to visualize:

Particle density heatmaps

CO₂ & occupancy correlation

HVAC load and contamination trends

Zone-wise differential pressure stability

 Model Training

Offline Training:
train_and_stream.py — trains baseline regression/LSTM models.

Online Prediction:
train_online.py — updates models with streaming data.

Model artifacts (.keras, .pkl, .npy) are stored locally in models/ (ignored by git).

 Data & Security

.env holds InfluxDB / MQTT credentials (never committed).

Large datasets (sensor_all_data.csv, raw_v2.csv) remain under data/raw/ (git-ignored).

Repository adheres to clean separation of code vs data vs models.

Supports future API token rotation and secure BMS integration.

 Tech Stack
Category	Technology
Programming	Python 3.10 + TensorFlow + PyTorch
Data Pipeline	Kafka / MQTT / InfluxDB 2.0
Dashboard	Grafana
Deployment	Docker / AWS ECS
Source Control	GitHub / Git Actions
🧪 Future Roadmap

 Integrate with HVAC BMS for closed-loop recommendations.

 Deploy edge inference models on Raspberry Pi / Jetson.

 Add anomaly detection and predictive maintenance modules.

 Publish research whitepaper + provisional patent.

📜 License

This repository is released under the MIT License unless otherwise specified.
