import streamlit as st
from transformers import AutoProcessor, BarkModel
from IPython.display import Audio

st.title("Text to Speech Converter")
st.write("Enter text, and the Bark model will generate speech!")

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

input_text = st.text_area("Enter text here", "Hello, I wanted to let you know that Bark is amazing!")

if st.button("Generate Speech"):
    inputs = processor(input_text, voice_preset=voice_preset)
    
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    
    sample_rate = model.generation_config.sample_rate
    st.audio(audio_array, format="audio/wav", sample_rate=sample_rate)

st.write("You can enter any text in the box above and click 'Generate Speech' to hear the generated audio.")
st.write("Note: Depending on the length of the text, it may take a moment to generate the speech.")
