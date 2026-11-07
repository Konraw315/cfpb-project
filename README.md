# 🏦 CFPB Complaint Response Predictor

[![Live Demo](https://img.shields.io/badge/🤗_Live_Demo-Hugging_Face-FFD21E)](https://huggingface.co/spaces/Konraw315/cfpb-response-predictor)
[![Model](https://img.shields.io/badge/🤗_Model-Hugging_Face-FFD21E)](https://huggingface.co/Konraw315/cfpb-predictor)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

An end-to-end machine learning project that predicts how a financial company is likely to respond to a consumer complaint submitted to the Consumer Financial Protection Bureau.

The project demonstrates the complete ML workflow: exploratory data analysis, data preprocessing, baseline model comparison, hyperparameter optimization, experiment tracking, model serving with FastAPI, and deployment of an interactive Streamlit application on Hugging Face Spaces.

The final solution uses an XGBoost classifier trained on historical CFPB complaint data. A public demo is available on Hugging Face, while the repository also includes a FastAPI prediction service for local or API-based use.

## 🌐 Live demo

👉 **Try the application:** [CFPB Response Predictor on Hugging Face Spaces](https://huggingface.co/spaces/Konraw315/cfpb-response-predictor)

The deployed Streamlit application downloads the trained model from the separate [Konraw315/cfpb-predictor](https://huggingface.co/Konraw315/cfpb-predictor) repository and performs inference directly in the Space.

> Hugging Face may put an inactive Space to sleep, so the first launch can require a short cold start.

## 🎯 What this project demonstrates

- 📊 Exploratory data analysis on a real-world financial dataset
- 🧹 Data cleaning and preprocessing
- 🤖 Comparison of multiple classification models
- ⚙️ Hyperparameter optimization with Optuna
- 🧪 Experiment tracking with MLflow
- 🚀 Model deployment with FastAPI
- 🖥️ Interactive prediction interface with Streamlit
- 🤗 Application and model deployment with Hugging Face Spaces and Hub

## 🗜️ Data preprocessing and storage optimization

The original raw dataset was stored as a CSV file with a size of approximately **7.0 GB**.

After:

* selecting only the relevant features,
* cleaning and preprocessing the data,
* assigning appropriate data types to each column,
* and converting the dataset from CSV to the Parquet format,

the final dataset size was reduced to approximately **48 MB**.

This represents a reduction of more than **99%**, significantly improving storage efficiency, loading speed, and memory usage during model training.

## 📊 Model performance

The evaluated models were compared using the macro-averaged F1 score, which assigns equal importance to every target class.

| Model                             | Configuration                                                | Macro F1 score |
| --------------------------------- | ------------------------------------------------------------ | -------------: |
| Dummy Classifier                  | Baseline                                                     |           0.17 |
| Balanced Random Forest Classifier | Vanilla                                                      |           0.38 |
| CatBoost Classifier               | Vanilla                                                      |           0.49 |
| XGBoost Classifier                | Vanilla                                                      |           0.04 |
| **XGBoost Classifier**            | **Balanced sample weights and Optuna hyperparameter tuning** |      **0.60** |

The final XGBoost model achieved the best result. Applying balanced sample weights and tuning the model's hyperparameters with Optuna increased the macro F1 score to **0.60**, substantially outperforming all baseline configurations.



## 🔄 Application flow

### Public Hugging Face demo

```text
Complaint details
        │
        ▼
Streamlit interface
        │
        ▼
Model downloaded from Hugging Face Hub
        │
        ▼
XGBoost classification model
        │
        ▼
Predicted company response and confidence
```

### Local API architecture

```text
Complaint details
        │
        ▼
Streamlit interface
        │
        ▼
FastAPI prediction endpoint
        │
        ▼
XGBoost classification model
        │
        ▼
Predicted company response and confidence
```

## 🧰 Tech stack

`Python` · `Pandas` · `DuckDB` · `scikit-learn` · `XGBoost` · `Optuna` · `MLflow` · `FastAPI` · `Streamlit` · `Hugging Face Spaces` · `Hugging Face Hub`

## 🧠 Model inputs

The model uses the following complaint attributes:

- `Product`
- `Issue`
- `Company`
- `Submitted via`
- `Consumer consent provided?`

## 📁 Repository structure

```text
cfpb-project/
├── EDA/                       # Exploratory data analysis notebooks
├── api/
│   └── api.py                 # FastAPI prediction service
├── data/
│   ├── raw/raw.py             # Raw data conversion
│   ├── interim/interim.py     # Intermediate preprocessing
│   ├── processed/processed.py # Final feature selection
│   └── categories/cat_save.py # Category export for the UI
├── models_code/
│   ├── dummy.py               # Dummy baseline
│   ├── vanila_brf.py          # Balanced Random Forest baseline
│   ├── vanila_catboost.py     # CatBoost baseline
│   ├── vanila_lr.py           # Logistic Regression baseline
│   ├── vanila_xgboost.py      # XGBoost baseline
│   └── xgboost_optuna.py      # Tuned XGBoost pipeline
├── streamlit_app.py           # Streamlit frontend
├── requirements.txt
├── LICENSE
└── README.md
```

## ⚙️ Installation

Clone the repository.

Create and activate a virtual environment and install the dependencies.

## 🏋️ Train the model

The trained model is not included in the repository.

Generate the required model artifacts locally:

```bash
python models_code/xgboost_optuna.py
```

The training pipeline creates:

```text
models/xgb_cfpb.ubj
models/label_encoder.joblib
models/cfpb_predictor.joblib
```

The current training configuration uses CUDA by default. For CPU training, change the XGBoost `device` setting from `cuda` to `cpu`.

## 🚀 Run the FastAPI service

Start the API from the repository root:

```bash
uvicorn api.api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## 🖥️ Run the Streamlit application

Open a second terminal and run:

```bash
streamlit run streamlit_app.py
```

The Streamlit application sends prediction requests to the FastAPI backend.

## 📈 Training workflow

1. Load the processed CFPB dataset
2. Clean the target column
3. Encode the response category
4. Create a stratified train-validation split
5. Compare baseline classifiers
6. Tune XGBoost with Optuna
7. Track experiments with MLflow
8. Train the final model
9. Save the model artifacts locally

## 🙋 Author

**Konrad Walkowiak**
