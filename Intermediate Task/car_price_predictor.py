import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # Added for saving the model artifact
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # Added RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Added mean_squared_error

# 1. Load data
csv_file = "car.csv"
if not os.path.exists(csv_file):
    print(f"Error: {csv_file} missing from current directory.")
    exit()

print("Loading dataset...")
df = pd.read_csv(csv_file)
print(f"Dataset loaded. Total rows: {len(df)}")

# Remove any accidental leading/trailing whitespaces from the header titles
df.columns = df.columns.str.strip()


# 2. Preprocessing & Feature Engineering
print("Processing data features...")

# Convert manufacturing year into vehicle age using the current year (2026)
df['Car_Age'] = 2026 - df['Year']

# Remove identifier text and the original year column
df = df.drop(columns=['Car_Name', 'Year'])

# Convert categorical columns (text labels) into 0 and 1 dummy variables
df_encoded = pd.get_dummies(df, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

# Cast True/False boolean column types to 1/0 integers for the model
bool_cols = df_encoded.select_dtypes(include=['bool']).columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)


# 3. Train-Test Split
# Separate the independent features from the target column (Selling_Price)
X = df_encoded.drop(columns=['Selling_Price'])
y = df_encoded['Selling_Price']

# Split dataset into 80% training and 20% validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. Model Training (Upgraded with Hyperparameter Tuning)
print("Optimizing Random Forest hyperparameters via RandomizedSearchCV...")

# Define the hyperparameter grid space
param_distributions = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Base model instance
rf_base = RandomForestRegressor(random_state=42)

# Set up Cross-Validated Random Search Optimization
rf_tuned = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_distributions,
    n_iter=15,
    cv=3,
    random_state=42,
    n_jobs=-1,
    scoring='r2'
)

# Run hyperparameter search training
rf_tuned.fit(X_train, y_train)
best_rf = rf_tuned.best_estimator_

print(f"✅ Optimal Strategy Selected: {rf_tuned.best_params_}")


# 5. Model Evaluation (Upgraded with MSE & RMSE)
# Run predictions using the newly optimized best model configuration
preds = best_rf.predict(X_test)

# Calculate model performance metrics
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

print("\n--- Model Evaluation ---")
print(f"R2 Score: {r2 * 100:.2f}%")
print(f"MAE: {mae:.3f} Lakhs")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f} Lakhs")

# Create a quick dataframe to preview true vs predicted outputs side by side
results_preview = pd.DataFrame({
    'Actual': y_test, 
    'Predicted': np.round(preds, 2)
}).head(10)
print("\nSample Predictions:")
print(results_preview.to_string())


# 6. Feature Importance Ranking
# Extract relative impact weights from the best selected estimator
importances = best_rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns, 
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df.to_string(index=False))


# 7. Plotting Results
plt.figure(figsize=(8, 6))

# Plot actual vs predicted prices as a scatter chart
plt.scatter(y_test, preds, color='blue', alpha=0.6, edgecolors='black')

# Draw a 45-degree reference line representing an exact 1:1 prediction match
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')

# Chart labels and formatting
plt.title('Actual vs Predicted Car Prices (Tuned Model)')
plt.xlabel('Actual Price (Lakhs)')
plt.ylabel('Predicted Price (Lakhs)')
plt.grid(True, linestyle=':', alpha=0.6)

# Save the plot image locally
plt.savefig('price_prediction_plot.png')
plt.close()
print("\nPlot successfully saved as 'price_prediction_plot.png'")


# 8. Save Model Checkpoint for Deployment (New Section)
print("\nSaving structural deployment checkpoints...")
deployment_package = {
    'model': best_rf,
    'features': X.columns.tolist()
}
joblib.dump(deployment_package, "car_price_predictor.pkl")
print("🎉 Model saved successfully as 'car_price_predictor.pkl'!")