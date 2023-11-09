import streamlit as st
from transformers import AutoProcessor, BarkModel
from IPython.display import Audio

st.title("Text to Speech Converter")
st.write("Enter text, and the Bark model will generate speech!")

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

input_text = st.text_area("Enter text here", "Hello, I wanted to let you know that Bark is amazing!")

