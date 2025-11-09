#define metrics like R², RMSE, MAE
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error , median_absolute_error, r2_score
import shap
import pandas as pd


def evaluate_model(model, X_test, y_test):
    y_predicted = model.predict(X_test)
    y_error = np.abs(y_test - y_predicted)

    metrics = dict()
    
    # From dummy
    metrics['Q1'] = np.percentile(y_error, 25)
    metrics['MedAE'] = np.percentile(y_error, 50)
    metrics['Q3'] = np.percentile(y_error, 75)

    # Standard
    metrics['MAE'] = mean_absolute_error(y_test, y_predicted)
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_predicted))
    metrics['R2'] = r2_score(y_test, y_predicted)

    # From paper
    metrics['MAPE (%)'] = np.mean(np.abs((y_test - y_predicted) / y_test)) * 100
    metrics['MedAE (€)'] = median_absolute_error(y_test, y_predicted)

    # Pretty print (like dummy)
    print("\n" + "="*60)
    print(" " * 20 + "MODEL EVALUATION")
    print("="*60)
    for k, v in metrics.items():
        if 'MAPE' in k:
            print(f"{k:20}: {v:8.2f}")
        elif isinstance(v, float):
            print(f"{k:20}: {v:,.0f}")
        else:
            print(f"{k:20}: {v}")
    print("="*60)

    return metrics

def feature_i(model, X_train):
    model_fi = model.feature_importances_
    index = X_train.columns
    feature_importance = pd.Series(model_fi, index = index)
    feature_importance = feature_importance.sort_values(ascending=False)
    return feature_importance

def shap_initialise(model, X, n = 100):
    sample_index = np.random.choice(len(X), size = n, replace = False)
    X_sample = X.iloc[sample_index]
    explain = shap.TreeExplainer(model)
    shap_values = explain.shap_values(X_sample)
    return explain, shap_values, X_sample

def compute_shap_local(explain, shap_values, X_sample):
    #Local
    i = 0 #Random instance
    shap.force_plot(
        explain.expected_value,
        shap_values[i].values,
        X_sample.iloc[i],
        matplotlib = True
    )
    shap.plots.bar(shap_values[i])

def compute_shap_global(explain, shap_values, X_sample):
    #Global
    shap.summary_plot(shap_values, X_sample, plot_type="bar")
    shap.summary_plot(shap_values, X_sample, plot_type="dot")

def compute_shap_cluster(explain, shap_values, X_sample):
    shap.summary_plot(shap_values, X_sample, plot_type="bar", cluster = True)

