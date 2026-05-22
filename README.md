# AI/ML Internship Portfolio

Welcome to my portfolio repository! This collection tracks my journey through the internship, showcasing my growth from foundational computer vision tasks to advanced localized Large Language Model (LLM) deployment.

## 📊 Repository Summary

| Level | Task | Focus Area | Performance/Status |
| :--- | :--- | :--- | :--- |
| **Beginner** | [Image Classification](./Task1/) | CNNs, Transfer Learning | ✅ 97.30% Accuracy |
| **Intermediate** | [Car Price Prediction](./IntermediateTask/) | Random Forest Regressor | ✅ 96.00% R2 Score |
| **Advanced** | [TinyLlama Deployment](./AdvanceTask/) | LLM, Few-Shot Prompting | ✅ Completed |

---

## 🟢 Beginner: Image Classification
**Objective:** Build a robust CNN capable of classifying local images into three categories: Cars, Cats, and Dogs.

### 🛠️ Design & Optimization
* **Baseline:** Built a custom CNN from scratch. Faced overfitting issues due to small data limitations.
* **Upgrade:** Transitioned to **MobileNetV2** (pre-trained on ImageNet).
* **Techniques:** Frozen base layers, Data Augmentation (Random Flips/Rotations), and Dropout (0.3) to prevent memorization.

### 📊 Performance Metrics
> **Validation Accuracy: 97.30%** | **Training Accuracy: 100%**
* The performance curve shows an optimized learning path with the overfitting gap successfully closed.

---

## 🟡 Intermediate: Car Price Prediction
**Objective:** Estimate the selling price of used cars based on critical vehicle metrics.

### 🛠️ Design & Optimization
* **Feature Engineering:** Converted production year into a dynamic `Car_Age` metric (2026 - Year) to capture depreciation.
* **Processing:** Implemented One-Hot Encoding for categorical features (Fuel, Seller Type, Transmission).
* **Algorithm:** Utilized an **Ensemble Random Forest Regressor** (100 estimators) to map non-linear price dependencies.

### 📊 Performance Metrics
> **R-squared (R2) Score: 0.9600** | **Mean Absolute Error (MAE): 0.651 Lakhs**
* Diagnostics show a tightly packed linear correlation along the true vs. predicted regression diagonal.

---

## 🔴 Advanced: Localized LLM & Comparative Analysis
**Objective:** Implement and rigorously analyze TinyLlama-1.1B in a localized, hardware-constrained environment.

### 🛠️ Design & Optimization
* **Implementation:** Developed a Jupyter pipeline for model initialization, tokenization, and configuration.
* **Prompt Engineering:** Used **Few-Shot Prompting** to enforce strict output formatting and mitigate generation errors.
* **Research Focus:** Investigated the relationship between token generation, computational throughput, and creativity.

### 💡 Key Insights
* **Adaptability:** High performance in task-specific domains; showed limitations in open-ended creative generation.
* **Efficiency:** Validated that Small Language Models (SLMs) are a viable, low-latency alternative to cloud-based APIs.
* **Future Scope:** Plan to explore Parameter-Efficient Fine-Tuning (PEFT) for better domain adaptation.

---

## ⚙️ How to Navigate
Each folder contains its own `README.md`, Python scripts/notebooks, and visualization artifacts.
1. Navigate to the task directory (e.g., `/AdvanceTask/`).
2. Review the `README.md` for methodology.
3. Open the `.ipynb` or `.py` files to explore the code implementation.

---
*Internship Repository | Maintained by Deeksha K P*
