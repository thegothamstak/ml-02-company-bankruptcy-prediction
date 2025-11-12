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

## Installation Instructions (Coming Soon)

This section will include steps to:
- Set up the project environment
   - On MacOS or Linux run the following command to create the project directory structure
      - mkdir -p data/raw data/processed data/final reports models src experiments
- Install required dependencies  
- Configure paths and settings  

Expected update: Once model training scripts and configuration files are finalized.

---

## Usage Guide (Coming Soon)

This section will provide instructions to:
- Train and evaluate the model  
- Run predictions on new data  
- Visualize feature importance and model metrics  

Expected update: After model source code and CLI/notebook interfaces are implemented.

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

- Temporal modeling for early warning  
- Ensemble models combining tree-based and neural nets  
- Zero-shot risk scoring using financial text embeddings  
- Expand dataset to include other East Asian countries  

---

## Contributors

- Bakary Sanogo
- Doren Chan
- Oshinee Mendis
- Shripad Tak
