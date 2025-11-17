# Bankruptcy Risk Prediction Using Financial Ratios from Taiwanese Companies

## Project Description

This project develops a machine learning model to predict corporate bankruptcy using financial ratios from Taiwanese companies, based on data from the Taiwan Economic Journal (1999–2009). By analyzing 96 financial indicators, the model enables financial institutions, investors, and regulators to identify high-risk companies early and make informed decisions. The pipeline includes data preprocessing, model training, evaluation, interpretation, and optional deployment.

---

## Project Overview

### Business Case

Traditional bankruptcy assessments rely on manual analysis and static rules, which often miss subtle patterns in financial data. This project applies machine learning to historical financial ratios to predict bankruptcy risk, offering scalable, data-driven insights for financial stakeholders.

### Project Goal

Build a robust machine learning model that accurately predicts bankruptcy and identifies the most influential financial features contributing to insolvency.

### Success Criteria

To consider this project successful, the following outcomes should be achieved:

1. **Model Performance Success Criteria**

| Metric     | Competitive Threshold | Ideal Target |
|------------|-----------------------|--------------|
| F1-score   | >= 0.60               | >= 0.70      |
| Recall     | >= 0.70               | >= 0.80      |
| Precision  | >= 0.60               | >= 0.75      |
| ROC AUC    | >= 0.95               | >= 0.97      |

3. **Feature Insights**
   - Identify and explain key financial features using SHAP or feature importance

4. **Reproducible Pipeline**
   - Documented ML pipeline covering preprocessing, modeling, evaluation, and interpretation

5. **Collaborative Repository**
   - Version-controlled GitHub repo with issue tracking and clear structure

6. **User-Friendly Documentation**
   - Comprehensive README with project overview, dataset summary, modeling approach, and usage instructions

7. **Optional Deployment Interface**
   - CLI, notebook, or dashboard for real-time predictions (if time permits)

8. **Accessibility**
   - Clear setup and execution instructions for users to run the model and generate predictions

---

## Problem Statement

Bankruptcy prediction is challenging due to class imbalance — bankrupt companies represent only 3.23% of the dataset. This imbalance can bias models and obscure high-risk firms. The goal is to build a classifier that identifies bankruptcy risk with high precision and recall.

---

## Stakeholders and Business Impact

| Stakeholder         | Impact                                                                 |
|---------------------|------------------------------------------------------------------------|
| Investors & Lenders | Better credit decisions, reduced financial exposure                    |
| Employees           | Improved job security through early risk detection                     |
| Management          | Proactive mitigation strategies                                         |
| Policy Makers       | Data-driven financial safeguards                                        |
| Researchers         | Insights into financial risk modeling                                   |

---

## Dataset Summary

- **Source**: Taiwan Economic Journal via Kaggle  
- **Period**: 1999–2009  
- **Instances**: 6,819 companies  
- **Features**: 96 financial ratios (e.g., debt ratio, net profit margin, ROA)  
- **Target**: Binary label — 1 for bankrupt, 0 for non-bankrupt  
- **Class Distribution**: Bankruptcy cases represent approximately 3.23% of the dataset

---

## Project Structure

```
project-root
├── data
├──── raw         # Original dataset
├──── processed   # Cleaned and transformed data
├──── final       # Final dataset
├──── sql         # If data living in a database (most likely not need this!)
├── experiments   # A folder for experiments
├── models        # A folder containing trained models or model predictions - saved as .pkl
├── reports       # Generated EDA, PDF report, final presentation
├── src           # Project source code
├── README.md     # Project documentation
└── .gitignore    # Files to exclude from version control (e.g., large data files)
```

---

## Installation Instructions

Follow these steps to set up the project environment and install dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/thegothamstak/ml-02-company-bankruptcy-prediction.git
cd ml-02-company-bankruptcy-prediction
```

### 2. Create Directory Structure

Run the following command to create the required directories (on macOS/Linux/Windows):
```bash
mkdir -p data/raw data/processed data/final reports models src experiments
```

### 3. Set Up Environment Variables

Create a .env file in the project root with the following paths (adjust as needed):
```
DATA_RAW=./data/raw
DATA_PROCESSED=./data/processed
DATA_FINAL=./data/final
MODELS_DIR=./models
REPORTS_DIR=./reports
SRC_DIR=./src
EXPERIMENTS_DIR=./experiments
```

Install python-dotenv to load these variables:
```bash
pip install python-dotenv
```

### 4. Create a Virtual Environment

Use Python 3.9+ (as per the notebook's kernel).
```bash
python -m venv env
source env/bin/activate  # On macOS/Linux
# Or on Windows: env\Scripts\activate
```

### 5. Install Dependencies

Install the required packages using pip. You can run requirements.txt file and run pip install -r requirements.txt.
```
imbalanced-learn
joblib
jupyter
keras
matplotlib
numpy
pandas
python-dotenv
scikit-learn
scipy
seaborn
tensorflow
tqdm
xgboost
```

### 6. Download the Dataset

- Download data.csv from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction).
- Place it in data/raw/data.csv.

---

## Usage Guide (Coming Soon)

### 1. Running the Complete Pipeline

Open the notebook:
```bash
jupyter notebook BankruptcyML.ipynb
```
Then execute cells in order — the pipeline is sequential.

### 2. Data Loading

The notebook automatically loads the raw dataset from:
```bash
raw_dir = os.getenv("DATA_RAW")
df = pd.read_csv(os.path.join(raw_dir, "your_filename.csv"))
```
To use a different file, replace the filename in the notebook or drop it into the data/raw/ folder with the same name.

### 3. Data Preprocessing

The notebook automatically handles:

#### Feature scaling
Using :
```bash
StandardScaler()
```

#### Class imbalance
Using :
```bash
SMOTE(random_state=42)
```

#### Train-test split
Default 80/20 split.

No manual action required — just run the notebook cells.

### 4. Training the Neural Network Model

The notebook trains a Sequential MLP model, with:
- Dense(64)
- Dense(32)
- Dropout
- BatchNormalization
- Binary output layer

This step runs automatically when you execute the Model Training section of the notebook.

The model is saved as:
```bash
models/bankruptcy_model.pkl
```

### 5. Training the Neural Network Model

The notebook includes a prediction function like:
```bash
model.predict(new_data_scaled)
```

To run predictions:
1. Place new financial ratio data in a CSV file
2. Apply the same preprocessing steps (scaler + selected features)
3. Use the loaded model to generate predictions

You may update the notebook to accept a CSV path and output predictions into:
```bash
data/final/predictions.csv
```

### 6. Evaluating the Model

The notebook automatically computes:
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Precision-Recall curve

All evaluation plots are generated and visible inside the notebook.

### 7. Feature Importance & SHAP Analysis

Your notebook loads SHAP and generates:
- Summary plot
- Beeswarm plot

To regenerate:
```bash
explainer = shap.KernelExplainer(model.predict, X_sample)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

SHAP results help explain which financial ratios contribute most to bankruptcy predictions.

### 8. Saving & Loading Models

The notebook saves trained models to:
```bash
models/
```

You can reload the model as:
```bash
import joblib
model = joblib.load("models/bankruptcy_model.pkl")
```

---

## Modeling Approach

### Data Preprocessing

- Verified absence of nulls and placeholder values  
- Scaled features using StandardScaler  
- Addressed class imbalance using SMOTE  
- Performed feature selection using correlation matrix, tree-based importance, or PCA  

### Algorithms

| Model              | Notes                          |
|-------------------|----------------------------------|
| Logistic Regression | Interpretable, fast baseline    |
| Random Forest      | Robust, useful for feature importance |
| XGBoost / LightGBM | High accuracy, fast training     |
| MLP (Neural Net)   | Flexible, scalable with dropout and batch normalization |

### Evaluation Metrics

- Accuracy  
- Precision, Recall, F1-score  
- ROC-AUC  
- Confusion Matrix  
- PR Curve  

### Interpretability

- SHAP values  
- Feature importance plots  
- Threshold tuning for business needs  

---

## Results Summary

- Best-performing model identified  
- Evaluation metrics reported  
- Top predictive features explained  
- Achieved over 90% accuracy in identifying bankruptcy drivers  

---

## Limitations

- Dataset limited to Taiwanese companies  
- Data is dated (1999–2009)  
- Class imbalance affects model sensitivity  
- Deep learning models require more data and tuning  

---

## Future Work

- Add Explainability (SHAP) to show why a company is predicted as risky
- Shift from binary label → risk probability bands (Low / Medium / High)
- Expand dataset with more years or external economic indicators
- Develop a stakeholder dashboard for decision-making

---

## Contributors

- Bakary Sanogo
- Doren Chan
- Oshinee Mendis
- Shripad Tak

---

## Video links

- Bakary Sanogo
https://drive.google.com/file/d/1j_hJ0z0-NDSiyfh4GAezYrdVhBdBzMki/view?usp=drive_link

- Doren Chan
https://drive.google.com/file/d/1hgYuVI3_iTWpyKcyIYAI-axEb5SVntrm/view?usp=sharing

- Oshinee Mendis
https://drive.google.com/drive/folders/1UxK-7ZkIXTyFYdYGV4Ld_39qgdlDIqt2?usp=sharing

- Shripad Tak
https://drive.google.com/drive/folders/1S4hyG_nD3p53pIgwYjS7TEHRM9RkaXSj