# 🚀 Task 1: Image Classification

## 📌 Project Objective
The goal of this project is to build a robust Convolutional Neural Network (CNN) capable of classifying local images into three distinct categories: **Cars**, **Cats**, and **Dogs**, using a custom dataset.

## 🛠️ Design & Optimization Journey
1. **Custom CNN Baseline:** Built an initial network from scratch. While training accuracy scaled up well, validation accuracy suffered from severe overfitting due to the limitations of learning complex object patterns from a small custom local dataset.
2. **Transfer Learning Upgrade:** To resolve the generalization gap, the pipeline was upgraded to utilize a pre-trained **MobileNetV2** architecture (trained on the 1.4-million-image ImageNet dataset).
3. **Architecture Details:**
   - Frozen base feature extraction layers to preserve global knowledge weights.
   - Added structural Data Augmentation (Random Flips & Rotations).
   - Attached a custom classification dense head with a Dropout layer ($0.3$) to eliminate memorization paths.

## 📊 Final Performance Metrics
- **Training Accuracy:** `1.0000` (**100%**)
- **Validation Accuracy:** `0.9730` (**97.30%**)
- **Training Epochs:** 10

The final performance curve shows a perfectly optimized learning path where validation metrics closely track training metrics, completely eliminating the overfitting gap.

## 📂 Artifacts Included
- `task1.py`: Complete executable training script utilizing TensorFlow/Keras.
- `accuracy_plot.png`: Generated performance chart displaying the final accuracy scores.