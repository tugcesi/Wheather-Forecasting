import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the weather data from a CSV file
data = pd.read_csv('weather_data.csv')  # Adjust the path as necessary

# 1. Time Series Plot
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Temperature'], label='Temperature')
plt.title('Time Series of Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('time_series_temperature.png')
plt.show()

# 2. Distribution Plot
plt.figure(figsize=(10, 5))
sns.histplot(data['Temperature'], bins=30, kde=True)
plt.title('Temperature Distribution')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.savefig('temperature_distribution.png')
plt.show()

# 3. Parameters Plot
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Humidity'], label='Humidity')
plt.plot(data['Date'], data['WindSpeed'], label='Wind Speed')
plt.title('Weather Parameters Over Time')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('parameters_over_time.png')
plt.show()

# 4. Correlation Heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Weather Parameters')
plt.savefig('correlation_heatmap.png')
plt.show()

# 5. Monthly Average Plot
monthly_avg = data.resample('M', on='Date').mean()
plt.figure(figsize=(10, 5))
plt.plot(monthly_avg.index, monthly_avg['Temperature'], label='Monthly Avg Temperature')
plt.title('Monthly Average Temperature')
plt.xlabel('Month')
plt.ylabel('Average Temperature')
plt.legend()
plt.tight_layout()
plt.savefig('monthly_average_temperature.png')
plt.show()

# 6. Scatter Plot
plt.figure(figsize=(10, 5))
sns.scatterplot(x=data['Temperature'], y=data['Humidity'])
plt.title('Scatter Plot of Temperature vs Humidity')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.savefig('scatter_temperature_humidity.png')
plt.show()

# 7. Box Plot
plt.figure(figsize=(10, 5))
sns.boxplot(data['Temperature'])
plt.title('Box Plot of Temperature')
plt.ylabel('Temperature')
plt.savefig('box_plot_temperature.png')
plt.show()