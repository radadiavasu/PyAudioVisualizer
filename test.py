import streamlit as st
from pydub import AudioSegment
import os
import pandas as pd

def get_audio_details(file_path):
    audio = AudioSegment.from_file(file_path)
    duration = len(audio) / 1000  # Duration in seconds
    sample_rate = audio.frame_rate
    channels = audio.channels
    frame_count = audio.frame_count()
    frame_rate = audio.frame_rate
    
    return {
        "File Name": os.path.basename(file_path),
        "Duration (s)": duration,
        "Sample Rate (Hz)": sample_rate,
        "Channels": channels,
        "Frame Count": frame_count,
        "Frame Rate (frames/sec)": frame_rate
    }

def process_directory(directory_path):
    audio_details = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(('.mp3', '.wav', '.flac', '.ogg', '.aac')):
                file_path = os.path.join(root, file)
                details = get_audio_details(file_path)
                audio_details.append(details)
    return audio_details

# Streamlit app
st.title("Audio Files Details Extractor")

uploaded_dir = st.text_input("Enter the directory path containing audio files:")

if st.button("Process Directory"):
    if os.path.isdir(uploaded_dir):
        audio_details_list = process_directory(uploaded_dir)
        
        if audio_details_list:
            st.write(f"Found {len(audio_details_list)} audio files.")
            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(audio_details_list)
            # Display the DataFrame in table format
            st.dataframe(df)
        else:
            st.write("No audio files found in the specified directory.")
    else:
        st.write("Please enter a valid directory path.")
with st.expander("Detail Explanation"):
    st.markdown("""
        **Speaker Identification**
        - **About**: This animation shows the evolution of the spectrogram over time.
        - **Why Use**: Useful for observing changes in the spectral content of the audio signal dynamically.
        - **When to Use**: Use this for detailed time-frequency analysis and to observe transient events in the audio.
        - **Main Purpose**: To visualize how the frequency components of the audio signal evolve over time in an animated format.

        [Spectrogram Animation](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Animated_spectrogram_example.gif/800px-Animated_spectrogram_example.gif)
    """)