import streamlit as st
from transformers import AutoProcessor, BarkModel
from IPython.display import Audio

# About App 
st.title("Text to Speech Converter")
st.write("Enter text and select a speaker for speech generation!")

# Load model
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

# Voice presets options
voice_presets = {
    "English Speaker 1": "v2/en_speaker_0",
    "English Speaker 2": "v2/en_speaker_1",
    "English Speaker 3": "v2/en_speaker_2",
    "English Speaker 4": "v2/en_speaker_3",
    "English Speaker 5": "v2/en_speaker_6",
    "Spanish Speaker": "v2/es_speaker_0",
    "French Speaker": "v2/fr_speaker_0",
    "Italian Speaker": "v2/it_speaker_0",
    "Japanese Speaker": "v2/ja_speaker_0",
    "Chinese Speaker": "v2/zh_speaker_0",
}

input_text = st.text_area("Enter text here", "Hello, I wanted to let you know that Bark is amazing!")

# Selecting voice preset
selected_voice_preset = st.selectbox("Select Speaker", list(voice_presets.keys()))

if st.button("Generate Speech"):

    voice_preset = voice_presets[selected_voice_preset]

    inputs = processor(input_text, voice_preset=voice_preset)
    
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    
    sample_rate = model.generation_config.sample_rate
    st.audio(audio_array, format="audio/wav", sample_rate=sample_rate)

st.write("Click 'Generate Speech' to hear the generated audio.")
st.write("Note: Depending on the length of the text, it may take a moment to generate the speech.")
