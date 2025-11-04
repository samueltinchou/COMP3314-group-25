#define metrics like R², RMSE, MAE
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
import shap


def evaluate_model(model, X_test, y_test):
    y_predicted = model.predict(X_test)
    y_error = y_test - y_predicted

    metrics = dict()
    metrics['Q1'] = np.percentile(np.abs(y_error), 25)
    metrics['MedAE'] = np.percentile(np.abs(y_error), 50)
    metrics['Q3'] = np.percentile(np.abs(y_error), 75)

    metrics['MAE'] = mean_absolute_error(y_test, y_predicted)
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_predicted))
    metrics['MSLE'] = mean_squared_log_error(y_test, y_predicted)
    metrics['R2'] = r2_score(y_test, y_predicted)

    return metrics

def shap_initialise(model, X, n):
    sample_index = np.random.choice(len(X), size = n, replace = False)
    X_sample = X.iloc[sample_index]
    explain = shap.TreeExplainer(model)
    shap_values = explain(X_sample)
    return explain, shap_values, X_sample

def compute_shap_local(model, X_test, n = 100):
    explain, shap_values, X_sample = shap_initialise(model, X_test, n)

    #Local
    i = 0 #Random instance
    shap.force_plot(
        explain.expected_value,
        shap_values[i].values,
        X_sample.iloc[i],
        matplotlib = True
    )
    shap.plots.bar(shap_values[i])

def compute_shap_global(model, X_test, n = 100):
    explain, shap_values, X_sample = shap_initialise(model, X_test, n)

    #Global
    shap.summary_plot(shap_values, X_sample)
    shap.summary_plot(shap_values, X_sample, plot_type="dot")
