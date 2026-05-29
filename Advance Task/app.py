import gradio as gr
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load the Model and Tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading {model_id}...")

# Determine device (Use CPU for free Hugging Face tier, GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype
).to(device)
print("Model loaded successfully.")

# 2. Define the Inference Function
def generate_response(prompt, max_tokens, temperature):
    start_time = time.time()
    
    # Format the prompt using the chat template (similar to your notebook)
    formatted_prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=int(max_tokens), 
            temperature=float(temperature),
            do_sample=True if float(temperature) > 0 else False
        )
        
    latency = time.time() - start_time
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (remove the prompt from the output)
    final_response = response.split("<|assistant|>\n")[-1].strip()
    
    return final_response, f"{latency:.2f} seconds"

# 3. Create the Gradio Interface
demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(lines=4, placeholder="Enter your prompt here...", label="User Prompt"),
        gr.Slider(minimum=10, maximum=512, step=10, value=100, label="Max Tokens"),
        gr.Slider(minimum=0.0, maximum=1.5, step=0.1, value=0.7, label="Temperature (Creativity)")
    ],
    outputs=[
        gr.Textbox(label="Model Output"),
        gr.Textbox(label="Inference Latency")
    ],
    title="🦙 TinyLlama Localized Deployment",
    description="An exploration of open-weight Small Language Models (SLMs). This interface demonstrates real-time inference latency and contextual generation capabilities."
)

if __name__ == "__main__":
    demo.launch()