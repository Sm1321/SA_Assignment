# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df1 = pd.read_csv('Transactional_data_retail_01.csv')
df1['InvoiceDate'] = pd.to_datetime(df1['InvoiceDate'], format='%d %B %Y')
df2 = pd.read_csv('Transactional_data_retail_02.csv')
df2['InvoiceDate'] = pd.to_datetime(df2['InvoiceDate'], format='%d-%m-%Y')
df = pd.concat([df1, df2])

# Calculate Revenue
df['Revenue'] = df['Quantity'] * df['Price']

# Group and aggregate data
df_grouped = df.groupby(['StockCode', 'InvoiceDate']).agg({
    'Quantity': 'sum',
    'Revenue': 'sum'
}).reset_index()

# Identify top 10 products by quantity sold
top_products_by_quantity = df_grouped.groupby('StockCode')['Quantity'].sum().nlargest(10).index.tolist()

# Streamlit app layout
st.title('Demand Forecasting App with Prophet')
st.write('This app forecasts demand for the top products based on historical sales data.')

# Dropdown for stock codes
selected_stock_code = st.selectbox('Select Stock Code', top_products_by_quantity)

# Function to apply Prophet forecasting and plot results
def apply_prophet_and_plot(df, product_code):
    # Prepare product data
    product_data = df[df['StockCode'] == product_code][['InvoiceDate', 'Quantity']]
    product_data.columns = ['ds', 'y']  # Rename columns for Prophet
    product_data.dropna(inplace=True)

    model_prophet = Prophet()
    model_prophet.fit(product_data)

    future = model_prophet.make_future_dataframe(periods=15)  # Forecast for 15 weeks
    forecast = model_prophet.predict(future)

    # Plotting historical and forecast demand
    plt.figure(figsize=(12, 6))
    plt.plot(product_data['ds'], product_data['y'], label='Historical Demand', color='orange')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Demand', color='blue')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.5)
    plt.title(f'Historical and Forecast Demand for Product {product_code}')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid()
    st.pyplot(plt)

    # Calculate errors
    train_size = len(product_data)
    actual = product_data['y'].values
    predicted = forecast['yhat'].values[:train_size]

    errors = actual - predicted
    test_errors = forecast['yhat'].values[train_size:] - forecast['yhat'].values[train_size:]  # Errors for the forecast period

    # Error metrics
    st.write(f'Mean Absolute Error (MAE): {np.mean(np.abs(errors)):.2f}')
    st.write(f'Mean Squared Error (MSE): {mean_squared_error(actual, predicted):.2f}')
    st.write(f'Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(actual, predicted)):.2f}')

    # Plot error histograms
    plt.figure(figsize=(12, 6))
    
    # Histogram for training errors
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=20, color='gray', alpha=0.7, edgecolor='black')
    plt.title('Error Distribution (Training Set)')
    plt.xlabel('Error')
    plt.ylabel('Frequency')

    # Histogram for forecast errors
    plt.subplot(1, 2, 2)
    plt.hist(test_errors, bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Error Distribution (Forecast Set)')
    plt.xlabel('Error')
    plt.ylabel('Frequency')

    plt.tight_layout()
    st.pyplot(plt)

    return forecast  # Return forecast data if needed

# When button is pressed
if st.button('Forecast'):
    forecast_data = apply_prophet_and_plot(df_grouped, selected_stock_code)

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Forecasting Done On the given Stock ID")
