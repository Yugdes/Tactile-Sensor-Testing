import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import joblib

# Added imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
data = pd.read_csv('combined_sensor_data.csv')


# Feature selection
features = data[['Flex Resistance 1(Ohms)', 'Flex Resistance 2(Ohms)', 'Flex Resistance 3(Ohms)']]
force_target = data['Force (N)']
location_target = data['Location']


# Apply noise reduction (moving average filter)
features_smoothed = features.rolling(window=3, min_periods=1).mean()  # Smoothing with a window size of 3


# Normalize features and force target using MinMaxScaler
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features_smoothed)
force_target_normalized = scaler.fit_transform(force_target.values.reshape(-1, 1)).flatten()


# Encode the location target
label_encoder = LabelEncoder()
location_target_encoded = label_encoder.fit_transform(location_target)


# Train-test split
X_train, X_test, y_force_train, y_force_test, y_loc_train, y_loc_test = train_test_split(
    features_normalized, force_target_normalized, location_target_encoded, 
    test_size=0.2, random_state=42
)


# Reshape data for LSTM (samples, timesteps, features)
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


# Build the LSTM model for force prediction
lstm_model = Sequential([
    LSTM(256, activation='relu', return_sequences=True,
         input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    LSTM(128, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output layer for regression (force prediction)
])


# Compile the LSTM model
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])


# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)


# Train the LSTM model
history = lstm_model.fit(
    X_train_lstm, 
    y_force_train, 
    epochs=50, 
    batch_size=16, 
    validation_split=0.2, 
    callbacks=[early_stopping], 
    verbose=1
)


# Evaluate the LSTM model
force_predictions = lstm_model.predict(X_test_lstm).flatten()
force_mse = mean_squared_error(y_force_test, force_predictions)
print(f"Force Prediction (LSTM) - MSE: {force_mse}")


# Build and train Random Forest model for location prediction
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf_model.fit(X_train, y_loc_train)


# Evaluate the Random Forest model
location_predictions = rf_model.predict(X_test)
location_accuracy = accuracy_score(y_loc_test, location_predictions)
print(f"Location Prediction (Random Forest) - Accuracy: {location_accuracy}")


# Save the LSTM model
lstm_model.save('final_lstm_force_model.h5')


# Save the Random Forest model
joblib.dump(rf_model, 'final_random_forest_location_model.pkl')


# Save the scaler and label encoder
joblib.dump(scaler, 'final_scaler.pkl')
joblib.dump(label_encoder, 'final_label_encoder.pkl')


print("Models and preprocessing tools have been saved.")


# ─────────────────────────────────────────────────────────────
# ADDITIONAL PLOTTING SECTION
# ─────────────────────────────────────────────────────────────

# 1) LSTM Training/Validation Curves (from 'history')
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title("LSTM Training and Validation Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()


# 2) Final MSE as a simple bar
plt.figure(figsize=(4, 4))
plt.bar(["Test MSE"], [force_mse], color='steelblue')
plt.title("Final LSTM MSE on Test Set")
# Provide a little buffer above the bar
plt.ylim(0, max(force_mse * 1.2, 0.01))  
plt.tight_layout()
plt.show()


# 3) Residual Analysis for LSTM Force
residuals = y_force_test - force_predictions

# (A) Residual vs. Predicted Force
plt.figure(figsize=(6, 5))
sns.scatterplot(x=force_predictions, y=residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residual vs. Predicted Force (Normalized Scale)")
plt.xlabel("Predicted Force (Normalized)")
plt.ylabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.show()

# (B) Histogram of Residuals
plt.figure(figsize=(6, 5))
sns.histplot(residuals, bins=50, kde=True, color='purple')
plt.title("Distribution of Residuals (Force, Normalized)")
plt.xlabel("Residual (N, normalized)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# 4) Confusion Matrix for Location
cm = confusion_matrix(y_loc_test, location_predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Location Prediction)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

print("\nAll additional plots generated.")
