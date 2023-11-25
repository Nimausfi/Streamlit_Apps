import streamlit as st
from PIL import Image
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion model
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

# Hardcoded parameters for image generation
guidance_scale = 4.5
temperature = 1.0
top_k = 50

# Streamlit app
st.title("Text to Image App")

# Text input for user prompt
prompt = st.text_area("Enter your text prompt:", height=5)

# Function to generate and display the image
def generate_image():
    with autocast(device):
        # Adjust parameters for image generation
        image = pipe(prompt, guidance_scale=guidance_scale, temperature=temperature, top_k=top_k).images[0]

    # Save the image
    image_path = 'generatedimage.png'
    image.save(image_path)

    # Display the image
    st.image(image, caption="Generated Image", use_column_width=True)

    # Provide a download link for the generated image
    st.markdown(f"Download the generated image: [Download]({image_path})")

# Button to trigger image generation
if st.button("Generate"):
    generate_image()
