#define metrics like R², RMSE, MAE
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

def evaluate_model(model, X_test, y_test):
    y_predicted = model.predict(X_test)
    y_error = y_test - y_error

    metrics = dict()
    metrics['Q1'] = np.percentile(np.abs(y_error), 25)
    metrics['MedAE'] = np.percentile(np.abs(y_error), 50)
    metrics['Q3'] = np.percentile(np.abs(y_error), 75)

    metrics['MAE'] = mean_absolute_error(y_test, y_predicted)
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_predicted))
    metrics['MSLE'] = mean_squared_log_error(y_test, y_predicted)
    metrics['R2'] = r2_score(y_test, y_predicted)

    return metrics
    