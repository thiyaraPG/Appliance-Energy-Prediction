# Energy Consumption Forecasting

## Overview

This project predicts appliance energy consumption using machine learning and deep learning models. 

The complete pipeline includes:

- Data preprocessing  
- Feature engineering  
- Baseline model development  
- Deep learning model training  
- Model evaluation  
- Hyperparameter optimization  
- Model saving and inference  

---

## Setup Instructions

### 1. Download the Project

Download or clone this repository to your computer.

### 2. Create virtual environment and activate it
```bash
python -m venv venv
```
Windows -> 
```bash
venv\Scripts\activate
```
macOS/Linux ->
```bash
source venv/bin/activate
```

### 3. Install dependencies

Make sure Python (version 3.9 or above) is installed.
```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Option 1: Run Using Jupyter Notebook (Recommended)

1. Open terminal in the project folder
2. Start Jupyter Notebook:

```bash
jupyter notebook
```

3. Open the notebook file (`EDA_and_Modeling.ipynb`)
4. Click **"Run All Cells"**

This will execute the full pipeline step-by-step

---

### Option 2: Run Using Python Script

1. Open terminal in the project folder
2. Navigate to the source folder:

```bash
cd src
```

3. Run the training pipeline:

```bash
python train.py
```

This will:

* preprocess data
* perform feature engineering
* train baseline models
* train deep learning models
* evaluate performance
* optimize the best model
* save the final trained model

4. Using the Trained Model (Inference)

```bash
cd src
python predict.py
```
This will:

* load trained model
* load scalers and selected features
* prepare input sequences
* generate predictions
* display Actual vs Predicted values
---

## Project Outputs

After running the project:

Data Outputs:
* data/processed/energy_data_clean.csv
* data/processed/train_data.csv
* data/processed/test_data.csv
* data/processed/selected_features_data.csv

Model Outputs:
* models/trained_model.h5
* models/scaler_X.pkl
* models/scaler_y.pkl
* models/selected_features.pkl

---

## Notes

* Jupyter Notebook is recommended for step-by-step understanding
* Python script is recommended for quick execution
* Run train.py before predict.py (first time only)