# 🌌 World Away: Exoplanet Hunting with AI
An interactive web application built using Streamlit and TensorFlow to classify exoplanets based on observational data. This tool leverages artificial intelligence to analyze exoplanet features, identify potential candidates, and provides users with an intuitive interface to explore, visualize, and export results.

---
## Table of Contents
- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)
- [References](#references)

## Demo
## 📹 [Project Demo](https://drive.google.com/file/d/1Ru-vpFIdBUKzNg9witRMb7ESzi6cZ1bB/view?usp=drive_link)

## Features
- Download Kepler-like data from NASA (attempted mapping to KOI fields).
- Generate synthetic KOI-style dataset for demo and testing.
- Interactive scatter plots and feature visualizations (Plotly).
- Configurable DNN training (layers, units, dropout, batch-norm, learning rate).
- Single-sample prediction with probability bars.
- Batch prediction from CSV, with downloadable results.
- Model analysis: confusion matrix, classification report, training history, and feature correlations.
- Save/load model (Keras `.h5`) and metadata (`joblib`).
## Tech Stack
- Python 3.8+
- Streamlit (UI)
- TensorFlow / Keras (model)
- scikit-learn (preprocessing, metrics)
- pandas, numpy (data)
- plotly (interactive charts)
- matplotlib, seaborn (optional)
- joblib (metadata save/load)
- requests (data download)
## Repository Structure

```
.
├─ app.py                  
├─ requirements.txt        # required packages
├─ README.md
├─ models/
│  └─ exoplanet_model.h5 
├─ data/
│  └─ k2_dataset.csv
│  └─ kepler_dataset.csv
│  └─ TESS_dataset.csv
└─ docs/
   └─ App screens
   └─ Media

```
## Quick Start
### Prerequisites
- Python 3.8 or above
- Required libraries: `streamlit`,`numpy`, `pandas`, `plotly`, `tensorflow`, `requests`.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MohamedSaeed130/NasaSpaceAppsChallenge-World-Away-Hunting-for-Exoplanets-with-AI.git
   cd NasaSpaceAppsChallenge-World-Away-Hunting-for-Exoplanets-with-AI
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the app locally with:
   ```bash
   python -m streamlit run App.py

## Acknowledgements

- NASA Kepler & TESS missions and the NASA Exoplanet Archive.
- NASA Space Apps Challenge organizers.
- Open-source libraries: TensorFlow, scikit-learn, Streamlit, Plotly, pandas.

## Contact

Mohamed Saeed Mohamed — mohamedsaidabusamaha@gmail.com  
GitHub: https://github.com/MohamedSaeed130  
LinkedIn: https://linkedin.com/in/mohamedabusamaha7


## References
- NASA Exoplanet Archive: [https://exoplanetarchive.ipac.caltech.edu](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/)

   



