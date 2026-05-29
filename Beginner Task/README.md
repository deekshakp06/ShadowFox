# 🚀 Task 1: Intelligent Image Tagging System

### 📌 Project Objective
The goal of this project is to build and deploy a production-ready, cloud-hosted Image Tagging System. Leveraging deep convolutional neural networks, the system processes user-uploaded images and dynamically classifies them into three distinct real-world categories: **Cars**, **Cats**, and **Dogs**.

---

### 🛠️ Design & Optimization Journey

* **Transfer Learning Foundation:** Utilizing a pre-trained **MobileNetV2** backbone (trained on the 1.4-million-image ImageNet dataset) as a highly expressive feature extractor to compensate for a lean custom training dataset.
* **Embedded Preprocessing Pipeline:** Integrated a mathematical `Rescaling(1./127.5, offset=-1)` layer directly into the core model architecture. This permanently resolved local-vs-production environment preprocessing mismatches and vectorized raw image input arrays on the fly.
* **Resolution & Structural Optimization:** Upgraded baseline dimensions to MobileNetV2's native **224x224** resolution to recover granular pixel data (such as texture variations and hard edges). 
* **Regularization & Stability:** Implemented aggressive data augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation`) alongside a robust classification head featuring **BatchNormalization** to stabilize training steps and an upgraded **Dropout (0.4)** threshold to eliminate memorization paths.

---

### 🌐 Real-World Production Deployment
Moving beyond restrictive `localhost` constraints, the system is fully deployed to production cloud infrastructure using **Hugging Face Spaces**. 

* **Interface:** Powered by a clean, user-friendly **Gradio** web application wrapper.
* **Backend:** Driven by a cloud-optimized `tensorflow-cpu` pipeline designed for rapid inference, minimal memory consumption, and 24/7 global availability.
* **Live Production Link:** https://huggingface.co/spaces/deekshakp/Image-Classifier

---

### 📂 Technical Artifacts Included

* **`task1.py`** – Complete, end-to-end model training, data augmentation, and performance evaluation pipeline.
* **`app.py`** – Lightweight production script orchestrating cloud-side Gradio UI inference logic.
* **`requirements.txt`** – Cloud-optimized manifest declaring server-side environment dependencies.
* **`my_model.keras`** – Saved, production-ready deep learning architecture weights.
* **`accuracy_plot.png`** – Clear visual analytics tracking training and validation convergence curves.
