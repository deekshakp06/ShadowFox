# AI/ML Internship Portfolio

Welcome to my portfolio repository! This collection tracks my journey through the internship, showcasing my growth from foundational computer vision tasks to advanced localized Small Language Model (SLM) deployment and web application production.

## 📊 Repository Summary

| Level | Task Link | Focus Area | Deployment / Status |
| :--- | :--- | :--- | :--- |
| **Beginner** | [Image Classification](./Beginner%20Task/) | CNNs, Transfer Learning, MobileNetV2 | ✅ 97.30% Accuracy |
| **Intermediate** | [Car Price Prediction](./Intermediate%20Task/) | Random Forest Regressor, Gradio | ✅ 96.00% R2 Score (Live App) |
| **Advanced** | [TinyLlama Deployment](./Advance%20Task/) | SLMs, Few-Shot Prompting, Latency Profiling | ✅ Local & Cloud Production Live |

---

## 🟢 Beginner: Image Classification
**Objective:** Build a robust CNN capable of classifying images into three categories: Cars, Cats, and Dogs.

### 🛠️ Design & Optimization
* **Baseline:** Built a custom CNN from scratch. Faced overfitting issues due to small data limitations.
* **Upgrade:** Transitioned to **MobileNetV2** (pre-trained on ImageNet).
* **Techniques:** Frozen base layers, Data Augmentation (Random Flips/Rotations), and Dropout (0.3) to prevent memorization.

### 📊 Performance Metrics
> **Validation Accuracy: 97.30%** | **Training Accuracy: 100%**
* The performance curve shows an optimized learning path with the overfitting gap successfully closed.

---

## 🟡 Intermediate: Car Price Prediction
**Objective:** Estimate the selling price of used cars based on critical vehicle metrics and deploy an interactive user web application.

### 🛠️ Design & Optimization
* **Feature Engineering:** Converted production year into a dynamic `Car_Age` metric (2026 - Year) to capture depreciation.
* **Processing:** Implemented One-Hot Encoding for categorical features (Fuel, Seller Type, Transmission).
* **Algorithm:** Utilized an **Ensemble Random Forest Regressor** (100 estimators) to map non-linear price dependencies.
* **Interface Deployment:** Wrapped the model using **Gradio** into an interactive UI for instant price estimations.

### 📊 Performance Metrics
> **R-squared ($R^2$) Score: 0.9600** | **Mean Absolute Error (MAE): 0.651 Lakhs**
* Diagnostics show a tightly packed linear correlation along the true vs. predicted regression diagonal.

---

## 🔴 Advanced: Localized LLM & Comparative Analysis
**Objective:** Implement, optimize, and rigorously analyze TinyLlama-1.1B in a localized, hardware-constrained environment and push to cloud production.

### 🛠️ Design & Optimization
* **Implementation:** Developed a Jupyter pipeline for model initialization, tokenization, and configuration.
* **Prompt Engineering:** Used **Few-Shot Prompting** to enforce strict output formatting boundaries and mitigate generation errors.
* **Real-World Deployment:** Developed a production web dashboard utilizing **Gradio** and deployed it live to **Hugging Face Spaces**.
* **Data Privacy & Ethics:** By running inference locally and in standalone spaces, the design eliminates data leakage risks inherent in third-party cloud APIs.

### 💡 Key Insights
* **Adaptability:** High performance in task-specific domains when guided by few-shot anchors; showed limitations in open-ended creative generation.
* **Efficiency:** Validated that Small Language Models (SLMs) are a highly viable, fast alternative to heavy cloud-based APIs for latency-sensitive applications.
* **Live App Link:** https://huggingface.co/spaces/deekshakp/TinyLlama-Local-Deployment

---

## ⚙️ How to Navigate
Each folder contains its own comprehensive `README.md`, Python production scripts, Jupyter notebooks, and visualization plots.
1. Navigate to a specific task directory (e.g., `/Advance Task/` or `/Intermediate Task/`).
2. Review the dedicated task `README.md` for deep methodology breakdowns.
3. Open the `.ipynb` files or run the `app.py` files to explore the code implementation and web interfaces locally.

---
*Internship Repository | Maintained by Deeksha K P*
