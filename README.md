# **Weather Forecasting Application**

### **Overview**

This application provides real-time weather forecasts using a machine learning model, with an interactive front end built in React.js and a backend API built in FastAPI. Users can view predictions for temperature, rain probability, wind conditions for a selected date range.

---

## **Table of Contents**

1. [Frontend (React.js)](#frontend-reactjs)
2. [Backend (FastAPI)](#backend-fastapi)
3. [Project Setup and Installation](#project-setup-and-installation)
4. [Usage](#usage)
5. [Available Endpoints](#available-endpoints)
6. [Technology Stack](#technology-stack)

---

## **Frontend (React.js)**

### **1. Prerequisites**

- **Node.js** (v14 or later) and **npm** installed.

### **2. Installation**

1. Navigate to the `frontend` directory.
   ```bash
   cd frontend
   ```
2. Install dependencies.
   ```bash
   npm install
   ```

### **3. Environment Configuration**

Ensure the `BASE_URL` for the backend API is correctly set (e.g., `http://127.0.0.1:8000`).

### **4. Running the Application**

```bash
npm start
```

This will start the frontend server at `http://localhost:3000`.

### **5. Key Features**

- **Date Selection**: Users can choose a date range, restricted to future dates.
- **Weather Display**: Real-time weather information is displayed through charts and cards.
- **Interactive Visualizations**: Uses Recharts for responsive data visualization, including:
  - **Temperature Trends**
  - **Rain Probability**
  - **Wind Status**

### **6. Project Structure**

- `src/`: Main source folder.
  - `components/`: Contains UI components like `WeatherSummary`, `RainProbability`, etc.
  - `utils/`: Utility functions for formatting dates and data handling.

---

## **Backend (FastAPI)**

### **1. Prerequisites**

- **Python 3.8+** and **pip**.

### **2. Installation**

1. Navigate to the `backend` directory.
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment.

   ```bash
   python -m venv venv

   source venv/bin/activate  # On Linux

   venv\Scripts\activate # On Windows
   ```

3. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

### **3. Environment Configuration**

Ensure all necessary configurations, such as file paths are set in the code.

### **4. Running the Application**

```bash
uvicorn main:app --reload
```

This starts the FastAPI server at `http://127.0.0.1:8000`.

### **5. Key Features**

- **Data Update**: Manually update weather data.
- **Forecasting**: Provides weather forecasts based on user-selected dates.
- **Error Handling**: Manages invalid date ranges or empty data responses with informative error messages.

### **6. Project Structure**

- `main.py`: Contains API endpoints, model loading, and data handling functions.
- `model/`: Stores the `var_model2.pkl` for forecasting.
- `data/`: Directory for storing weather data CSV files.

---

## **Project Setup and Installation**

### **1. Install Frontend and Backend Dependencies**

Navigate to both frontend and backend directories, following the installation instructions provided.

---

## **Usage**

1. Start the backend server with FastAPI.
2. Start the frontend server with React.
3. Open the frontend in a browser at `http://localhost:3000`.
4. Use the date picker to select a date range, view weather forecasts, and explore interactive charts.

---

## **Available Endpoints**

### **Backend (FastAPI)**

1. **POST /forecast**

   - **Description**: Retrieves forecasted weather data.
   - **Parameters**: `start_date`, `end_date`.
   - **Response**: JSON with temperature, rain probability, wind speed.

2. **PUT /update-data**

   - **Description**: Updates the CSV data with the latest available weather data.
   - **Response**: Success message with the latest date.

3. **GET /temperature-advisory**

   - **Description**: Returns dates with extreme temperatures based on thresholds.
   - **Parameters**: `high_threshold`, `low_threshold`.
   - **Response**: JSON with dates and temperature values.

4. **GET /rain-probability**

   - **Description**: Calculates rain probability based on precipitation data.
   - **Parameters**: `rain_sum`.
   - **Response**: Probability percentage.

5. **GET /wind-advisory**

   - **Description**: Advises if wind speed exceeds a specified threshold.
   - **Parameters**: `threshold`.
   - **Response**: Dates and wind speed data.

6. **GET /temperature-range**
   - **Description**: Returns the temperature range (max - min) for each date.
   - **Response**: JSON with date and temperature range values.

---

## **Technology Stack**

### **Frontend**

- **React.js**: JavaScript framework for building user interfaces.
- **Recharts**: Data visualization library for React.
- **react-datepicker**: Date picker for selecting date ranges.
- **Bootstrap**: CSS framework for styling components.

### **Backend**

- **FastAPI**: Python framework for building APIs.
- **pandas**: Data analysis library for data manipulation.
- **statsmodels**: Library for statistical models (VAR).
- **pickle**: Used to load the machine learning model.

---

## **Troubleshooting**

1. **Frontend Issues**: Ensure that the backend server is running and that the `API_BASE_URL` is correctly configured in React.
2. **Backend Issues**: If there are model-related issues, confirm that `var_model2.pkl` is correctly placed in the `model/` directory and `csv_file_path` points to the CSV file.

---

This README provides a complete guide to setting up and running this Weather Forecasting Application.
