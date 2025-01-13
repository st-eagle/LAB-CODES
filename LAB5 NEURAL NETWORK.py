import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Suppress TensorFlow informational logs (only show errors)
tf.get_logger().setLevel('ERROR')

# Step 1: Data Preprocessing

# Load your dataset (replace the file path with your actual file path)

dataset_path = 'synthetic_traffic_data - synthetic_traffic_data.csv'
traffic_data = pd.read_csv(dataset_path)

# Convert 'timestamp' column to datetime format
traffic_data['timestamp'] = pd.to_datetime(traffic_data['timestamp'])

# Step 2: Time-Series Data Handling
traffic_data['hour'] = traffic_data['timestamp'].dt.hour
traffic_data['day_of_week'] = traffic_data['timestamp'].dt.dayofweek

# Step 3: Feature Engineering (Lag Features)
# Create lag features to capture historical traffic data (e.g., traffic flow from the previous hour)
traffic_data['traffic_flow_lag1'] = traffic_data['traffic_flow'].shift(1)
traffic_data = traffic_data.dropna()  # Drop the first row with NaN values after creating lag

# Step 4: Normalization of Features
# Normalize traffic flow and weather data (temperature, humidity, rain)
scaler = MinMaxScaler()
scaled_data = traffic_data[['traffic_flow', 'temperature', 'humidity', 'rain', 'hour', 'day_of_week', 'traffic_flow_lag1']]
scaled_data = scaler.fit_transform(scaled_data)

# Step 5: Prepare Input Features and Target
X = scaled_data[:, 1:]  # All features except traffic flow
y = scaled_data[:, 0]   # Target variable (traffic flow)

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 7: Reshape for LSTM (3D input: [samples, time_steps, features])
X_train_lstm = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])

# Step 8: Build LSTM Model
model = Sequential()
model.add(Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))  # Explicit Input layer
model.add(LSTM(units=50, activation='relu'))
model.add(Dense(units=1))  # Output layer for regression (traffic flow prediction)
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 9: Train the Model
history = model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=1)

# Step 10: Model Evaluation
# Predict traffic flow on the test set
y_pred = model.predict(X_test_lstm)

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Step 11: Visualization

# Plot Training & Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Predicted vs Actual Traffic Flow
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Traffic Flow')
plt.plot(y_pred, label='Predicted Traffic Flow')
plt.title('Actual vs Predicted Traffic Flow')
plt.xlabel('Time (Hours)')
plt.ylabel('Traffic Flow')
plt.legend()
plt.show()

# Step 12: Weather Impact on Traffic Flow Visualization
plt.figure(figsize=(10, 6))
plt.scatter(traffic_data['temperature'], traffic_data['traffic_flow'], c=traffic_data['rain'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Rain (0: No, 1: Yes)')
plt.title('Temperature vs Traffic Flow with Rain Impact')
plt.xlabel('Temperature (°C)')
plt.ylabel('Traffic Flow')
plt.show()
