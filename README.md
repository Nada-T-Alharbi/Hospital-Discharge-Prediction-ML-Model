# Clinical Discharge Prediction System

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)
![Dataset](https://img.shields.io/badge/Dataset-MIMIC--IV--Demo-orange)

## Project Overview
This repository presents an end-to-end machine learning system for predicting hospital discharge risk using structured clinical data. The project emphasizes sound methodology, interpretability, and practical deployment through an interactive dashboard.

The goal is to estimate the likelihood of patient discharge within a 48-hour window, based on admission information, vital signs, and laboratory data.

Note:
The system is designed as a clinical decision support prototype to aid prioritization and monitoring, not as an autonomous decision-making tool.

---

## Dataset: MIMIC-IV Clinical Database (Demo)

The project utilizes the de-identified MIMIC-IV Demo dataset, which is suitable for open research and code sharing.

Data Domains:
- Hospital admissions and transfers
- Patient demographics
- Laboratory events
- ICU vital signs and charted events

Directory Structure:
```text
Dataset/
├── hosp/
│   ├── admissions.csv
│   ├── patients.csv
│   ├── labevents.csv
│   └── transfers.csv
└── icu/
    └── chartevents.csv
```
Dataset Notes:
- Only the demo version of MIMIC-IV is used.
- The dataset is fully de-identified.
- The full MIMIC-IV dataset requires credentialed access and is not included.
- Data is used strictly for research and model prototyping.

---

Model:
- Type: Supervised machine learning classifier
- Task: 48-hour discharge risk prediction
- Training Size: 245 hospital admissions
- Performance:
  - AUROC: 0.677
  - Test Accuracy: 65%
- Output: Risk stratification (High / Medium / Low)

---

Application:
An interactive Streamlit dashboard is provided for:
- Patient-level discharge risk prediction
- Visualization of clinical inputs
- Feature contribution analysis
- Real-time model inference

---

Disclaimer:
This project is intended for research and educational purposes only.
Model outputs support analysis and prioritization and must not be used for real-world clinical decision-making.
