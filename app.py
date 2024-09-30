from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

app = Flask(__name__)

# Load the trained CNN model
model = load_model('sales_forecast_cnn.h5')

# Load the data
file_path = 'NaivasSupermarketData.csv'
data = pd.read_csv(file_path)

# Preprocess the data
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Month'] = data['Order Date'].dt.to_period('M')
monthly_sales = data.groupby(['Category', 'Month'])['Sales'].sum().reset_index()
sales_pivot = monthly_sales.pivot(index='Month', columns='Category', values='Sales')

# Create a route for the index page
@app.route('/', methods=['GET', 'POST'])
def index():
    plot = None
    plot_category = None
    
    if request.method == 'POST':
        category = request.form['category']
        if category in sales_pivot.columns:
            # Prepare data for forecasting
            category_sales = sales_pivot[category].dropna().values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(category_sales.reshape(-1, 1))
            
            # Prepare data for CNN
            time_step = 3
            X, _ = create_dataset(scaled_data, time_step)
            X = X.reshape(X.shape[0], X.shape[1], 1)  # Adjust based on CNN input shape

            # Forecasting
            forecast = forecast_sales(model, X, scaler, time_step)
            
            # Plot the results
            plot = plot_forecast(monthly_sales, category, forecast)
            plot_category = category

    return render_template('index.html', plot=plot, plot_category=plot_category)

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
    return np.array(X), np.array(Y)

def forecast_sales(model, X, scaler, time_step):
    forecast = []
    X_future = X[-1].reshape(1, time_step, 1)  # Adjust for CNN
    for _ in range(12):  # Forecast for the next 12 months
        prediction = model.predict(X_future)
        forecast.append(prediction[0][0])
        X_future = np.append(X_future[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
    return scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

def plot_forecast(monthly_sales, category, forecast):
    plt.figure(figsize=(12, 8))
    historical_dates = monthly_sales['Month'].dt.to_timestamp()
    plt.plot(historical_dates[monthly_sales['Category'] == category], 
             monthly_sales[monthly_sales['Category'] == category]['Sales'], 
             label='Historical Sales', marker='o')

    last_month = historical_dates.iloc[-1]
    forecast_dates = pd.date_range(start=last_month + pd.DateOffset(months=1), periods=12, freq='M')

    plt.plot(forecast_dates, forecast, label='Forecast', linestyle='--', marker='o')
    plt.title(f'Sales Forecast for {category}')
    plt.ylabel('Sales')
    plt.xlabel('Month')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig('static/forecast_plot.png')
    plt.close()

    return 'forecast_plot.png'

if __name__ == '__main__':
    app.run(debug=True)
