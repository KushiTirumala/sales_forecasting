from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_models(X_train, y_train):
    """Train multiple regression models."""
    
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and print metrics."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "R2": r2}
        print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")
    return results
