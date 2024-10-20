from fastapi import FastAPI, Request, Query
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
import pickle
from statsmodels.tsa.api import VAR
from datetime import timedelta, date, datetime
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

app = FastAPI()

templates = Jinja2Templates(directory="templates")

var_model_path = 'model/var_model2.pkl'
with open(var_model_path, 'rb') as file:
    var_model_fitted = pickle.load(file)

daily_relevant_columns = [
    'date', 'temperature_max', 'temperature_min', 'apparent_temperature_max',
    'precipitation_sum', 'wind_speed_max'
]

csv_file_path = 'data/daily_cleaned_data.csv'

def update_csv_data():
    daily_data = pd.read_csv(csv_file_path)
    daily_data['date'] = pd.to_datetime(daily_data['date'])

    # Add 'year' column based on the 'date' column
    daily_data['year'] = daily_data['date'].dt.year

    last_date = daily_data['date'].max().date()
    current_date = date.today()

    if current_date > last_date:
        start_date = last_date + timedelta(days=1)
        end_date = current_date
        
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude=-37.8228&longitude=145.0389&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,apparent_temperature_max,precipitation_sum,wind_speed_10m_max"
        response = requests.get(url)
        data = response.json()
        
        if 'daily' in data:
            new_data = pd.DataFrame(data['daily'])
            new_data.rename(columns={
                'time': 'date', 
                'temperature_2m_max': 'temperature_max', 
                'temperature_2m_min': 'temperature_min', 
                'wind_speed_10m_max': 'wind_speed_max'
            }, inplace=True)
            new_data['date'] = pd.to_datetime(new_data['date'])
            new_data['year'] = new_data['date'].dt.year  # Add 'year' column to new data
            daily_data = pd.concat([daily_data, new_data], ignore_index=True)
            daily_data.to_csv(csv_file_path, index=False)
    return daily_data

def calculate_rain_probability(rain_sum):
    # Ensure rain_sum is not negative
    rain_sum = max(rain_sum, 0)
    # Scale the rain probability based on the rain sum
    # Assuming a scale from 0 to 100% probability
    if rain_sum == 0:
        return 0  # No rain probability if there's no rain
    elif rain_sum <= 2.5:
        return min((rain_sum / 2.5) * 50, 50)  # Light rain scaled up to 50%
    elif rain_sum <= 7.6:
        return min((rain_sum / 7.6) * 75, 75)  # Moderate rain scaled up to 75%
    else:
        return min((rain_sum / 20) * 100, 100)  # Heavy rain capped at 100%

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "FastAPI Example", "message": "Hello, FastAPI!"})

@app.get("/daily_temperature")
def getDailyTemperature(start_date: date = Query(...), end_date: date = Query(...)):
    today_date = date.today()
    if start_date < today_date:
        return "You can't Predict the Past"
    else:
        date_difference = (end_date - start_date).days + 1  # Number of days to forecast

        daily_data = update_csv_data()
        # Prepare the last observations for forecasting
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data.set_index('date', inplace=True)
        daily_data.fillna(method='ffill', inplace=True)  # Fill missing data if necessary

        # Forecast the next date_difference days
        lag_order = var_model_fitted.k_ar
        last_observations = daily_data.values[-lag_order:]
        forecasted_values = var_model_fitted.forecast(last_observations, steps=date_difference)

        # Generate forecast dates
        forecast_dates = pd.date_range(start=today_date, periods=date_difference, freq='D')

        # Create a DataFrame with the forecasted values
        forecast_df = pd.DataFrame(forecasted_values, index=forecast_dates, columns=daily_data.columns)

        # Ensure any negative values are set to zero
        forecast_df[forecast_df < 0] = 0

        # Calculate rain probability
        forecast_df['rain_probability'] = forecast_df['precipitation_sum'].apply(calculate_rain_probability)

        # Convert DataFrame to dictionary for JSON response and convert Timestamps to ISO format
        forecast_df.reset_index(inplace=True)
        forecast_df.rename(columns={'index': 'date'}, inplace=True)
        forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')  # Convert dates to string format

        # Filter data between start_date and end_date
        forecast_df = forecast_df[
            (forecast_df['date'] >= start_date.strftime('%Y-%m-%d')) &
            (forecast_df['date'] <= end_date.strftime('%Y-%m-%d'))
        ]

        forecast_dict = forecast_df.to_dict(orient='records')

        return JSONResponse(content={"forecast": forecast_dict})
