from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def get_model(name: str, seed: int = 42):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {name}")
