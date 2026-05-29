import gradio as gr
import tensorflow as tf
import numpy as np

# 1. Load the fine-tuned, high-performance model
# We set compile=False to bypass loading optimizer states since we are only predicting
model = tf.keras.models.load_model('my_model.keras', compile=False)

# 2. Strict alphabetical class mapping matching the Keras dataset structure
labels = ['car', 'cat', 'dog'] 

def predict_image(img):
    if img is None:
        return "Please upload an image."
        
    # Step A: Convert Gradio's numpy array input into a float32 tensor
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    
    # Step B: Resize image to 224x224 to match the new high-accuracy model input shape
    img_resized = tf.image.resize(img_tensor, (224, 224))
    
    # Step C: Add batch dimension to create the expected shape: [1, 224, 224, 3]
    # Note: We do NOT use preprocess_input here because the model's built-in Rescaling layer handles it!
    img_final = np.expand_dims(img_resized, axis=0)
    
    # Step D: Run inference and extract the array of raw probabilities
    predictions = model.predict(img_final, verbose=0)[0]
    
    # Step E: Return a dictionary mapping each class string to its floating-point probability
    return {labels[i]: float(predictions[i]) for i in range(len(labels))}

# 3. Build and customize the Gradio Web User Interface
demo = gr.Interface(
    fn=predict_image, 
    inputs=gr.Image(label="Upload Image (Car, Cat, or Dog)"), 
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="Intelligent Image Tagging System",
    description="Upload an image containing a car, cat, or dog. The deep learning system will process the visual features and automatically apply the appropriate category tags with probability scores."
)


if __name__ == "__main__":
    # Launch the local web server
    demo.launch()