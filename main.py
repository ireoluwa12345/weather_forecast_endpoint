from fastapi import FastAPI, Request, Query, HTTPException

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


templates = Jinja2Templates(directory="templates")

var_model_path = 'model/var_model2.pkl'
with open(var_model_path, 'rb') as file:
    var_model_fitted = pickle.load(file)

daily_relevant_columns = [
    'date', 'temperature_max', 'temperature_min', 'apparent_temperature_max',
    'precipitation_sum', 'wind_speed_max'
]

csv_file_path = 'data/daily_cleaned_data.csv'

class ForecastRequest(BaseModel):
    start_date: date
    end_date: date

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

    # Endpoint 1: Home
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    GET /
    Returns a welcome HTML message for the application.
    """
    return templates.TemplateResponse("index.html", {"request": request, "title": "SAAD Group Waather Project", "message": "Welcome to the Weather Forecasting API!"})


@app.post("/forecast")
def getDailyTemperature(requestData: ForecastRequest):
    today_date = date.today()
    start_date = requestData.start_date
    end_date = requestData.end_date
    if start_date < today_date:
        return "You can't Predict the Past"
    else:
        date_difference = (end_date - today_date).days + 1  # Number of days to forecast
        print(date_difference)

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
        print(forecast_df)

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

# Endpoint 4: Rain Probability Calculation
@app.get("/rain-probability")
def get_rain_probability(rain_sum: float = Query(..., description="Rainfall amount for calculating probability")):
    """
    GET /rain-probability
    Calculates and returns the probability of rain based on the provided rain sum.
    """
    probability = calculate_rain_probability(rain_sum)
    return JSONResponse(content={"data": probability, "message": "Rain probability calculated successfully."})


@app.get("/temperature-advisory")
def temperature_advisory(
    high_threshold: float = Query(30.0, description="High temperature threshold"),
    low_threshold: float = Query(5.0, description="Low temperature threshold")
):
    daily_data = update_csv_data()
    forecast = daily_data.to_dict(orient="records")  # Convert DataFrame to list of dictionaries
    
    advisories = []
    for day in forecast:
        # Check for advisory conditions
        try:
            if day["temperature_max"] > high_threshold or day["temperature_min"] < low_threshold:
                advisories.append({
                    "date": day["date"].strftime('%Y-%m-%d'),  # Convert Timestamp to string
                    "temperature_max": day["temperature_max"],
                    "temperature_min": day["temperature_min"]
                })
        except KeyError as e:
            print(f"Key error: {e}. Check the data structure.")
            return JSONResponse(content={"data": {}, "message": "Error occured, please try again."})

    return JSONResponse(content={"data": advisories, "message": "Temperature advisory data retrieved successfully."})


@app.get("/wind-advisory")
def wind_advisory(threshold: float = Query(20.0, description="Wind speed threshold for advisory")):
    forecast = update_csv_data()
    advisories = [
        {
            "date": day["date"],
            "wind_speed_max": day["wind_speed_max"]
        }
        for day in forecast if day["wind_speed_max"] > threshold
    ]
    return JSONResponse(content={"data": advisories, "message": "Wind advisory data retrieved successfully."})



@app.get("/temperature-range")
def temperature_range():
    daily_data = update_csv_data()
    forecast = daily_data.to_dict(orient="records")  # Convert DataFrame to list of dictionaries

    temperature_range = []
    for day in forecast:
        try:
            # Calculate temperature range
            temp_range = day["temperature_max"] - day["temperature_min"]
            
            # Check for NaN values in the calculated range
            if np.isnan(temp_range):
                temp_range = 0  # Replace NaN with 0 or any other desired default value
            
            temperature_range.append({
                "date": day["date"].strftime('%Y-%m-%d'),  # Ensure date is JSON serializable
                "temperature_range": temp_range
            })
        except KeyError as e:
            print(f"Key error: {e}. Check the data structure.")

    return JSONResponse(content={"data": {"temperature_range": temperature_range}, "message": "Temperature range data retrieved successfully."})



@app.put("/update-data")
def update_data():
    """
    PUT /update-data
    Manually triggers the update of CSV data.
    """
    try:
        updated_data = update_csv_data()
        latest_date = updated_data['date'].max().strftime('%Y-%m-%d')  # Get the latest date in the updated data
        
        return JSONResponse(
            content={
                "message": "Data successfully updated.",
                "latest_date": latest_date
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while updating data: {str(e)}")
