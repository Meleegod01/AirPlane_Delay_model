import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#show the data
data = pd.read_csv("airline_delay.csv")
print(data.describe())

# Handling missing values
data = data.fillna(0)

# Plot distribution of flights across top airports
top_airports = data['airport_name'].value_counts().nlargest(20).index
top_airports_data = data[data['airport_name'].isin(top_airports)]
plt.figure(figsize=(18, 8))  # Increase the width of the figure
sns.countplot(x='airport_name', data=top_airports_data, order=top_airports)
plt.title('Distribution of Flights Across Top 20 Airports')
plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate x labels and adjust font size
plt.xlabel('Airport Name')
plt.ylabel('Flight Count')
plt.show()

# Handle date column
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Seasonal variations in air traffic (example: month-wise analysis)
if 'date' in data.columns:
    data['month'] = data['date'].dt.month
    plt.figure(figsize=(12, 6))
    sns.countplot(x='month', data=data)
    plt.title('Monthly Air Traffic')
    plt.xticks(rotation=90)
    plt.show()

# Analysis of delays
plt.figure(figsize=(12, 6))
sns.histplot(data['arr_delay'], kde=True)
plt.title('Distribution of Arrival Delays')
plt.xticks(rotation=90)
plt.show()

# Analysis of cancellations
plt.figure(figsize=(12, 6))
sns.countplot(x='arr_cancelled', data=data)
plt.title('Cancelled Flights')
plt.xticks(rotation=90)
plt.show()

# Create the 'is_holiday' feature
holiday_list = [{'month': 1, 'year': 2020}, {'month': 12, 'year': 2020}]  # Example list
data['is_holiday'] = data.apply(lambda row: 1 if {'month': row['month'], 'year': row['year']} in holiday_list else 0, axis=1)

# Select relevant features
features = ['arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'security_ct', 'late_aircraft_ct', 'is_holiday']
target = 'arr_delay'

# Feature selection
plt.figure(figsize=(14, 10))  # Increase figure size
corr = data[features + [target]].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Split the data
X = data[features].fillna(0)  # Fill missing values with 0 or another strategy
y = data[target].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Standardize the data for clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(data_scaled)

# Visualize the clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x=data_scaled[:,0], y=data_scaled[:,1], hue=data['cluster'], palette='viridis')
plt.title('Cluster Analysis')
plt.xticks(rotation=90)
plt.show()

features = ['arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'security_ct', 'late_aircraft_ct', 'is_holiday']
data = data.fillna(0)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Function to update clustering
def update_clustering(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data_scaled)
    fig = px.scatter(x=data_scaled[:, 0], y=data_scaled[:, 1], color=data['cluster'].astype(str),
                     title=f'Cluster Analysis with {n_clusters} Clusters')
    fig.show()

# Display the initial plot
initial_clusters = 5
update_clustering(initial_clusters)

# Interactive input for clusters
while True:
    try:
        n_clusters = int(input("Enter the number of clusters (or type 'exit' to quit): "))
        update_clustering(n_clusters)
    except ValueError:
        print("Exiting...")
        break
    
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Arrival Delays')
plt.show()

import ipywidgets as widgets
# Function to update flight distribution plot
def update_flight_distribution(airport_count):
    top_airports = data['airport_name'].value_counts().nlargest(airport_count).index
    top_airports_data = data[data['airport_name'].isin(top_airports)]
    plt.figure(figsize=(18, 8))
    sns.countplot(x='airport_name', data=top_airports_data, order=top_airports)
    plt.title(f'Distribution of Flights Across Top {airport_count} Airports')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.xlabel('Airport Name')
    plt.ylabel('Flight Count')
    plt.show()

# Widget for flight distribution
airport_count_slider = widgets.IntSlider(min=1, max=20, step=1, value=10, description='Top Airports')
widgets.interact(update_flight_distribution, airport_count=airport_count_slider)

# Function to update seasonal variations plot
def update_seasonal_variations(selected_months):
    if 'month' in data.columns:
        filtered_data = data[data['month'].isin(selected_months)]
        plt.figure(figsize=(12, 6))
        sns.countplot(x='month', data=filtered_data, order=selected_months)
        plt.title('Monthly Air Traffic')
        plt.xticks(rotation=90)
        plt.show()

# Widget for seasonal variations
months_selector = widgets.SelectMultiple(
    options=range(1, 13),
    value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    description='Months'
)
widgets.interact(update_seasonal_variations, selected_months=months_selector)

# Function to update delay distribution plot
def update_delay_distribution(show_kde):
    plt.figure(figsize=(12, 6))
    sns.histplot(data['arr_delay'], kde=show_kde)
    plt.title('Distribution of Arrival Delays')
    plt.xticks(rotation=90)
    plt.show()

# Widget for delay distribution
kde_toggle = widgets.Checkbox(value=True, description='Show KDE')
widgets.interact(update_delay_distribution, show_kde=kde_toggle)

# Function to update clustering plot
def update_clustering(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data_scaled)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=data['cluster'], palette='viridis')
    plt.title('Cluster Analysis')
    plt.xticks(rotation=90)
    plt.show()

# Widget for clustering
clusters_slider = widgets.IntSlider(min=2, max=10, step=1, value=5, description='Clusters')
widgets.interact(update_clustering, n_clusters=clusters_slider)