import streamlit as st
from plotly import graph_objs as go
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

st.title("Favorita(Pichincha) Unit Sales Prediction App")

predict_input = st.text_input("Enter prediction horizon in days to predict", value="5", key="horizon")

try:
    predict_input = int(predict_input)
except:
    st.error("Please enter a valid integer.")
    st.stop()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = f"{BASE_DIR}/data/grouped_data.csv"
MODELS_DIR = f"{BASE_DIR}/models"
artifact = joblib.load(f"{MODELS_DIR}/xgb_sales_model.pkl")
pipe = artifact['model']
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = pd.concat([df,
           pd.DataFrame({'date': df['date'].max() + pd.to_timedelta(np.arange(0, predict_input+1), unit='d')})])
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
df['day'] = pd.to_datetime(df['date']).dt.day
df['dayofweek'] = pd.to_datetime(df['date']).dt.dayofweek

X = df[['year', 'month', 'day', 'dayofweek']]
y = df['unit_sales']
preds = pipe.predict(X)
df['Predictions'] = preds
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['unit_sales'], mode='lines+markers', name='Actual Unit sales',
                         line=dict(color='red')))
fig.add_trace(go.Scatter(x=df['date'], y=df['Predictions'], mode='lines+markers', name='Predicted Unit sales'))
fig.update_layout(title='Unit Sales(Pichincha) Prediction', xaxis_title='Date', yaxis_title='Unit Sales',
                  width = 1200)
st.plotly_chart(fig, width='stretch')