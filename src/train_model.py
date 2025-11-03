from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score


def train_random_forest(X,y):
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [500, 1000, 2500, 3000],
        'max_depth': [8, 16, 32, 64],
        'bootstrap': [True, False]
    }

    # Initialize model
    regressor = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Set up grid search with 5-fold CV
    grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        cv=5,
        scoring=make_scorer(r2_score),
        verbose=1
    )

    # Fit model
    grid_search.fit(X, y)

    # Extract best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params