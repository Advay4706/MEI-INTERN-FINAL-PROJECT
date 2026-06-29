# Z-Axis Thermal Error Analysis and Compensation Study using Machine Learning

## Overview

This project was developed during my internship at Mitsubishi Electric India Pvt. Ltd.

The objective of the project is to analyze thermal behaviour in a Vertical Machining Center (VMC) and predict Z-axis positioning error using Machine Learning. The project combines real machine temperature readings, position error measurements, exploratory data analysis, and regression models to estimate thermal error.

---

## Project Workflow

Raw Machine Data
        ↓
Data Parsing
        ↓
Temperature Dataset Creation
        ↓
Master Dataset Preparation
        ↓
Exploratory Data Analysis
        ↓
Machine Learning Model Training
        ↓
Model Evaluation
        ↓
Thermal Error Prediction

---

## Project Structure

```
MEI-INTERN-FINAL-PROJECT/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── graphs/
│
├── models/
│
├── scripts/
│   ├── 01_parse_position_log.py
│   ├── 02_create_temperature_dataset.py
│   ├── 03_master_dataset.py
│   ├── 04_data_analysis.py
│   └── 05_machine_learning.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Machine Learning Models

- Linear Regression
- Random Forest Regression

---

## Features Used

- RPM
- Spindle Temperature
- X-Axis Temperature
- Y-Axis Temperature
- Z-Axis Temperature
- Elapsed Time

Target Variable:

- Z-Axis Position Error

---

## Results

### Linear Regression

- MAE: 0.001857 mm
- RMSE: 0.002350 mm
- R² Score: 0.966608

### Random Forest

- MAE: 0.000197 mm
- RMSE: 0.000260 mm
- R² Score: 0.999591

Random Forest produced the highest prediction accuracy and was selected as the final model.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

## Author

Advay Aggarwal

Intern, Technical Development Team

Mitsubishi Electric India Pvt. Ltd.