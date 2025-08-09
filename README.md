# Daily Report App — OU Final Project

## 📌 Overview
This project is the **final project for the Open University**, focusing on **daily audience data analysis** using both **KMeans clustering** and **LSTM anomaly detection**.  
It processes data from MongoDB, applies machine learning models, and generates daily audience behaviour reports.

## 🚀 Features
- **Data Loading** from MongoDB/GridFS
- **KMeans Clustering** of audience viewing patterns
- **LSTM-based Anomaly Detection** for audience spikes and dips
- **Data Visualisation** of trends and anomalies
- **Daily Reporting** in both plot and table formats

## 📂 Repository Structure
File: daily-report_app
```
main.py                 # Entry point for running the daily report pipeline
loading.py              # Loads models and scalers from MongoDB GridFS
k_initial_analysis.py   # Initial KMeans data preparation and cleaning
k_cluster_analysis.py   # KMeans clustering and reporting
lstm_analysis.py        # LSTM anomaly detection and reporting
```

## 🛠 Requirements
- Python 3.9+
- MongoDB
- pandas, numpy, matplotlib
- scikit-learn
- tensorflow / keras
- pymongo, gridfs
- joblib

Install dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ How to Run

1. Ensure MongoDB is running and contains:
- PVF and IBT collections with audience data.
- Trained KMeans and LSTM models (and their scalers) stored in GridFS within the model_storage database.

2. Run:
```bash
python main.py
```

3. The script will:
   - Fetch audience data for the given date
   - Perform clustering and anomaly detection
   - Generate plots and tables for the daily report

## 📊 Example Output
- **Cluster Summary Report** (KMeans)
- **Weighted Audience Share Table**
- **Top Areas by Cluster**
- **Anomaly Detection Plot** (LSTM)
- **Anomalies Table** with audience spikes and dips


<div style="padding:64.95% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/1108743344?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media; web-share" referrerpolicy="strict-origin-when-cross-origin" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="Streamlit_movie"></iframe></div><script src="https://player.vimeo.com/api/player.js"></script>

## 📄 License
This project is for academic use as part of the Open University final project.
