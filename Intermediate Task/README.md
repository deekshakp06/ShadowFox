# 🚀 Task 2: Car Price Prediction

## 📌 Project Objective
The goal of this project is to build a high-performance predictive machine learning model to estimate the selling price of used cars based on critical vehicle metrics like showroom price, mileage, fuel source, and age.

## 🛠️ Design & Optimization Journey
1. **Feature Engineering Strategy:** Transformed raw attributes into structural signals. Converted the vehicle's production year into a dynamic `Car_Age` metric (2026 - Year) to explicitly capture asset depreciation. Irrelevant tracking data like model names were pruned.
2. **Categorical Feature Processing:** Integrated non-numeric operational features (Fuel Type, Seller Type, and Transmission Mechanism) into the mathematical pipeline via comprehensive One-Hot Encoding (`pd.get_dummies`).
3. **Algorithm Execution:** Selected an ensemble **Random Forest Regressor** with 100 estimator nodes to optimally map non-linear price dependencies, multi-variable correlation impacts, and structural market variances without over-fitting the localized data.

## 📊 Final Performance Metrics
- **R-squared ($R^2$) Score:** `0.9600` (**96.00%**)
- **Mean Absolute Error (MAE):** `0.651` Lakhs
- **Data Splitting Split:** 80% Training / 20% Testing

The final evaluation diagnostics display a tightly packed linear correlation along the true vs. predicted regression diagonal, validating that the ensemble trees interpret vehicle market values with precise accuracy.

## 📂 Artifacts Included
- `car_price_predictor.py`: Complete executable Python training pipeline.
- `car.csv`: Underlying structured dataset containing physical used car attributes.
- `price_prediction_plot.png`: Generated performance regression plot mapping true target data against model outputs.
