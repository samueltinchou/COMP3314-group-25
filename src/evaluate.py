#define metrics like R², RMSE, MAE
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error , median_absolute_error, r2_score
import shap
import pandas as pd
import matplotlib.pyplot as plt


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
    metrics['MSLE'] = mean_squared_log_error(y_test, y_predicted)

    # Pretty print (like dummy)
    print("\n" + "="*60)
    print(" " * 20 + "MODEL EVALUATION")
    print("="*60)
    for k, v in metrics.items():
        if 'MAPE' in k:
            print(f"{k:20}: {v:8.2f}")
        
        elif isinstance(v, float):
            print(f"{k:20}: {v:,.10f}")
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
    print("Start initialising")
    sample_index = np.random.choice(len(X), size = n, replace = False)
    X_sample = X.iloc[sample_index]
    explain = shap.TreeExplainer(model)
    shap_values = explain.shap_values(X_sample)
    return explain, shap_values, X_sample

def compute_shap_local(shap_values, X_sample):
    #Local
    i = 0 #Random instance
    shap.force_plot(
        shap_values[i].base_values,
        shap_values[i].values,
        X_sample.iloc[i],
        matplotlib = True
    )
    shap.plots.bar(shap_values[i])

def compute_shap_global(shap_values, X_sample, max_display=20):
    """
    Works whether shap_values is:
    - numpy array (regression or binary classification)
    - list of arrays (multi-class)
    - or output from XGBoost/LightGBM which sometimes wraps in list
    """
    
    # --- Convert shap_values to numpy array of shape (n_samples, n_features) ---
    if isinstance(shap_values, list):
        # Multi-class or wrapped output → take first class (or average later if needed)
        shap_vals = np.abs(shap_values[0]) if len(shap_values) == 1 else np.abs(shap_values[0])
    else:
        shap_vals = np.abs(shap_values)
    
    # Mean absolute SHAP value per feature (global importance)
    mean_abs_shap = shap_vals.mean(axis=0)
    
    # Get feature names
    feature_names = X_sample.columns if hasattr(X_sample, 'columns') else [f'feature_{i}' for i in range(X_sample.shape[1])]
    
    # Sort by importance
    idx = np.argsort(mean_abs_shap)[-max_display:][::-1]
    mean_abs_shap = mean_abs_shap[idx]
    feature_names = [feature_names[i] for i in idx]
    
    # --- Bar plot with values on bars ---
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(mean_abs_shap)-1, -1, -1), mean_abs_shap, color='steelblue')
    
    # Add text labels on the bars
    for i, (val, name) in enumerate(zip(mean_abs_shap, feature_names[::-1])):
        ax.text(val + max(mean_abs_shap)*0.01, len(mean_abs_shap)-1-i, 
                f'{val:.3f}', va='center', ha='left', fontweight='bold', fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names[::-1])
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', colors='gray')
    ax.set_xlabel("Mean |SHAP value| (Global Importance)")
    ax.set_title("Top Feature Importance (SHAP)")
    plt.tight_layout()
    plt.show()
    
    # --- Also show dot plot (beeswarm) ---
    print("\nBeeswarm summary plot:")
    shap.summary_plot(shap_values, X_sample, max_display=max_display)

def compute_shap_cluster(shap_values, X_sample):
    # Manual clustering for regression
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    import numpy as np
    import matplotlib.pyplot as plt

    abs_shap = np.abs(shap_values)
    dist = pdist(abs_shap.T, metric='correlation')
    Z = linkage(dist, method='average')
    order = leaves_list(Z)

    shap.summary_plot(
        shap_values[:, order],
        X_sample.iloc[:, order],
        feature_names=X_sample.columns[order],
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Summary (Clustered - Regression)")
    plt.tight_layout()
    plt.show()

