# 🚀 Task 2: Intelligent Car Price Valuation Engine

### 📌 Project Objective
The goal of this project is to build and deploy an end-to-end predictive machine learning model to accurately estimate the real-world resale value of used cars. By evaluating structural metrics—including original showroom price, vehicle usage metrics, fuel profile, and asset depreciation factors—the engine computes high-precision market valuations instantly.

---

### 🛠️ Design & Optimization Journey

* **Advanced Feature Engineering:** Transformed raw historical features into robust structural signals. Converted the vehicle's manufacturing timeline into a dynamic asset depreciation metric (`Car_Age = 2026 - Year`) to explicitly capture physical market decay. Irrelevant, high-cardinality metadata (such as vehicle model names) were completely pruned to eliminate dimensional noise.
* **Categorical Feature Processing:** Vectorized non-numeric operational attributes (`Fuel_Type`, `Seller_Type`, and `Transmission`) via comprehensive One-Hot Encoding (`pd.get_dummies`) and forced boolean variables into clean `1/0` integer layouts for mathematical compatibility.
* **Hyperparameter Optimization:** Upgraded the baseline architecture by implementing a cross-validated **`RandomizedSearchCV`** loop across a multi-dimensional parameter matrix. The search engine automatically determined the absolute best random forest structure:
  * `n_estimators`: 200 nodes (decision trees)
  * `max_features`: None (evaluating all features for split choices)
  * `min_samples_split`: 2
  * `min_samples_leaf`: 1
* **Feature Importance Analytics:** Production feature evaluation maps verified that a vehicle's original showroom cost (`Present_Price`) holds the dominant predictive weight (**88.69%**), followed directly by linear asset age decay (**5.90%**) and total usage mileage (**3.56%**).

---

### 📊 Final Performance Metrics

* **R-squared ($R^2$) Score:** `95.99%` (Explaining nearly 96% of unseen target variance)
* **Mean Absolute Error (MAE):** `0.630` Lakhs
* **Mean Squared Error (MSE):** `0.924`
* **Root Mean Squared Error (RMSE):** `0.961` Lakhs
* **Data Allocation Splits:** 80% Training Matrix / 20% Unseen Validation Control Set

*The resulting evaluation diagnostics display an exceptionally tight linear cluster configuration directly bounding the true-versus-predicted regression diagonal, proving high-precision estimation stability.*

---

### 🌐 Real-World Production Deployment
Moving beyond local terminal scripts, the hyperparameter-tuned model payload has been compiled and successfully deployed to live cloud infrastructure.

* **User Interface:** Powered by a clean, responsive web application wrapper built on **Gradio**.
* **Cloud Architecture:** Hosted 24/7 on **Hugging Face Spaces** using an optimized server runtime configuration.
* **Live Production Link:** https://huggingface.co/spaces/deekshakp/Car-Price-Predictor

---

### 📂 Technical Artifacts Included

* **`car_price_predictor.py`** – Automated data-cleaning pipeline, cross-validated hyperparameter optimization loops, and feature importance analysis.
* **`app.py`** – Server-side web deployment script orchestrating user interaction mapping and real-time model inference logic.
* **`car_price_predictor.pkl`** – Production checkpoint payload containing optimal estimator weights and structural feature layouts.
* **`requirements.txt`** – Environment blueprint file declaring server-side dependencies (`scikit-learn`, `pandas`, etc.).
* **`car.csv`** – Underlying structured dataset containing physical used car characteristics.
* **`price_prediction_plot.png`** – High-resolution diagnostic plot mapping true target values against model predictions.