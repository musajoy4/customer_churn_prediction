# customer_churn_prediction

## Project Overview

Customer churn is a critical challenge for telecom companies, leading to revenue loss and increased acquisition costs. This analysis uses machine learning to predict churn based on customer demographics, services, and billing information. Key models evaluated include Logistic Regression, Random Forest, XGBoost, and SVM, with hyperparameter tuning applied to the top performer (XGBoost) for optimal results.

### Key Features
- **Dataset**: IBM Telco Customer Churn (7043 samples, 21 features)
- **Target Variable**: Churn (Yes/No)
- **Models**: Logistic Regression, Random Forest, XGBoost, SVM
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Business Impact**: Quantifies potential savings from reducing churn
- **Visualizations**: Churn distribution, feature correlations, ROC curves, and more


## Dataset

The dataset is sourced from Kaggle's Telco customer churn
THE Data is loaded directly from the notebook using a Kaggle path. For local runs, download the CSV and update the path accordingly.


### Requirements
- Python 3.11+
- Jupyter Notebook

Install dependencies via pip:
```
pip install -r requirements.txt
```

**requirements.txt** (generated from the notebook):
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
joblib
```

### Notebook Sections
1. **Setup**: Library imports and dataset loading
2. **Data Inspection & Cleaning**: Handling missing values, type conversions
3. **Exploratory Data Analysis (EDA)**: Visualizations of churn rates, correlations, distributions
4. **Feature Engineering**: Encoding categorical variables, scaling numerics
5. **Model Training & Evaluation**: Compare multiple models with cross-validation
6. **Hyperparameter Tuning**: GridSearchCV on XGBoost
7. **Feature Importance**: Analyze key churn drivers (e.g., tenure, contract type)
8. **Business Impact**: Calculate ROI from improved predictions
9. **Model Saving**: Export for production use

## Results

### Model Performance (Optimized XGBoost)
| Metric       | Value  |
|--------------|--------|
| Accuracy     | 75.46% |
| Precision    | 49.27% |
| Recall       | 81.02% |
| F1-Score     | 61.28% |
| ROC-AUC      | 84.31% |

- **Best Model**: XGBoost (after tuning)
- **Key Insights**: Short tenure, month-to-month contracts, and high charges are top churn predictors
- **Visualizations** include confusion matrices, ROC curves, and feature importance plots.
