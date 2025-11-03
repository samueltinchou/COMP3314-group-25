from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, r2_score


def train_random_forest(X,y):
    #Step 1: Partition
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    #Step 2: Define the grids
    param_grid = {
        'n_estimators': [500, 1000, 2500, 3000],
        'max_depth': [8, 16, 32, 64],
        'bootstrap': [True, False]
    }

    #Step 3: Set up grid search and fir
    regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        cv=5, #fold
        scoring=make_scorer(r2_score),
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    #Step 4: Find the best model 
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    test_s = r2_score(y_test, best_model.predict(X_test))

    #According to the paper, n_estimators = 2500, max_depth = 32, bootstrap = F

    return best_model, best_params, test_s