import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

# 1. Configuration
DATASET_PATH = "./dataset"
IMG_SIZE = (224, 224)  # Upgraded to MobileNetV2's native resolution
BATCH_SIZE = 16        # Better gradient stability

# 2. Data Loading (Balanced and Shuffled)
print("--- Loading Datasets ---")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training", seed=42, 
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical', shuffle=True
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="validation", seed=42, 
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical', shuffle=True
)

CLASS_NAMES = train_dataset.class_names
print(f"Detected classes in alphabetical order: {CLASS_NAMES}\n")

# 3. Model Architecture
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights='imagenet'
)
base_model.trainable = False  # Keep the complex features frozen safely

model = models.Sequential([
    # Step A: Normalization Layer (Scales 0-255 pixels to [-1, 1] internally)
    layers.Rescaling(1./127.5, offset=-1, input_shape=(224, 224, 3)),
    
    # Step B: Advanced Data Augmentation (Prevents overfitting)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    
    # Step C: Pre-trained Base Feature Extractor
    base_model,
    layers.GlobalAveragePooling2D(),
    
    # Step D: High-Performance Classification Head
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),  # Keeps features scaled and balanced
    layers.Dropout(0.4),          # Breaks reliance on specific pixel patches
    layers.Dense(3, activation='softmax')
])

# Use a steady learning rate paired with the new normalization layers
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# 4. Training
print("--- Training Model ---")
EPOCHS = 25  # Increased epochs to let the stronger dense layers fully converge
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

# 5. Evaluation
print("\n--- Generating Final Classification Report ---")
y_true = []
y_pred = []

for images, labels in val_dataset:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

print("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# 6. Save Artifacts
model.save('my_model.keras')
print("✅ High-performance model saved to 'my_model.keras'")

# Create and save a clean performance plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('accuracy_plot.png')
print("✅ Performance graphs saved to 'accuracy_plot.png'")
print("\n🎉 Task 1 Training pipeline execution complete!")