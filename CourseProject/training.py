import joblib
from xgboost import XGBRegressor
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/grouped_data.csv")
MODELS_DIR = Path("models")


def main():

    artifact = joblib.load(MODELS_DIR / "xgb_sales_model.pkl")

    df = pd.read_csv(DATA_PATH)
    pipe = artifact['model']
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['day'] = pd.to_datetime(df['date']).dt.day
    df['dayofweek'] = pd.to_datetime(df['date']).dt.dayofweek
    X = df[['year', 'month', 'day', 'dayofweek']]
    y = df['unit_sales']

    pipe.fit(X, y)
    artifact['model'] = pipe
    artifact['version'] += 1
    joblib.dump(artifact, MODELS_DIR / "xgb_sales_model.pkl")

if __name__ == "__main__":
    main()