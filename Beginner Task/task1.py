# ==========================================
# STEP 1: IMPORT LIBRARIES & LOAD DATA
# ==========================================
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# Identify folder path dynamically
if os.path.exists("./Task 1/dataset"):
    DATASET_PATH = "./Task 1/dataset"
elif os.path.exists("./dataset"):
    DATASET_PATH = "./dataset"
else:
    print("❌ Error: Could not find the 'dataset' directory!")
    exit()

print(f"✅ Found dataset at: {DATASET_PATH}")

# MobileNetV2 expects larger images to extract rich features
IMG_SIZE = (128, 128)
BATCH_SIZE = 8

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# 🌟 ANTI-OVERFITTING 1: GENTLE AUGMENTATION
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

# Prefetch optimization for memory performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# ==========================================
# STEP 2: IMPORT PRE-TRAINED MOBILENETV2
# ==========================================
# We load Google's MobileNetV2 without its top layer, using weights trained on ImageNet
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)

# 🌟 ANTI-OVERFITTING 2: FREEZE THE BASE KNOWLEDGE
# This prevents your small dataset from messing up the pre-trained feature detectors
base_model.trainable = False

# ==========================================
# STEP 3: ASSEMBLE THE ACCURACY-BOOSTED MODEL
# ==========================================
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
    data_augmentation,
    
    # MobileNetV2 requires inputs scaled between [-1, 1]
    tf.keras.layers.Lambda(lambda x: tf.keras.applications.mobilenet_v2.preprocess_input(x)),
    
    base_model,
    
    # Global pooling compresses the complex features into a clean layout
    tf.keras.layers.GlobalAveragePooling2D(),
    
    # Classification head
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3), # Extra shield against overfitting
    tf.keras.layers.Dense(3, activation='softmax') # 3 classes: car, cat, dog
])

# ==========================================
# STEP 4: COMPILE AND RUN TRAINING
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Training Transfer Learning Model ---")
# Pre-trained models learn incredibly fast; 10 epochs is plenty to see high scores
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# ==========================================
# STEP 5: PLOT HIGH ACCURACY METRICS
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy Score')
plt.title('Transfer Learning Model Performance')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('accuracy_plot.png')
print("\n🎯 High-accuracy plot saved successfully as 'accuracy_plot.png'!")
plt.show()