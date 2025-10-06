### Single Model Training
```bash
# Train XGBoost (default)
python3 pipelines/training_pipeline.py

# Train Random Forest
python3 pipelines/training_pipeline.py --model_type random_forest

# Train Logistic Regression  
python3 pipelines/training_pipeline.py --model_type logistic_regression

# Train CatBoost
python3 pipelines/training_pipeline.py --model_type catboost
```

### Multiple Model Comparison
```bash
# Train and compare all models
python3 pipelines/training_pipeline.py --compare_all

# Or use the 'all' option
python3 pipelines/training_pipeline.py --model_type all
```

### Programmatic Usage
```python
from training_pipeline import training_pipeline, train_multiple_models

# Train single model
model, results = training_pipeline(model_type='xgboost')

# Train multiple models
results = train_multiple_models(['logistic_regression', 'random_forest'])

# Train with custom parameters
custom_params = {'n_estimators': 200, 'max_depth': 15}
model, results = training_pipeline(
    model_type='random_forest', 
    model_params=custom_params
)
```

## Test Results

All models have been successfully tested:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| XGBoost | 0.7921 | 0.6209 | 0.5177 | 0.5646 |
| Random Forest | 0.7842 | 0.6542 | 0.4886 | 0.5594 |
| Logistic Regression | 0.7906 | 0.6080 | 0.5083 | 0.5537 |
| CatBoost | 0.7516 | 0.5189 | 0.7405 | 0.6102 |

## Output Files

Models are automatically saved to:
- `artifacts/models/xgboost_telco_churn_analysis.joblib`
- `artifacts/models/random_forest_telco_churn_analysis.joblib`
- `artifacts/models/logistic_regression_telco_churn_analysis.joblib`
- `artifacts/models/catboost_telco_churn_analysis.joblib`
