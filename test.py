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
        - Identifying speakers from audio data is a common task in the field of speech processing. One approach is to extract relevant features from the audio, such as spectrograms, MFCCs (Mel-Frequency Cepstral Coefficients), or other representations, 
          and then use machine learning algorithms, such as neural networks or classifiers, to identify speakers based on these features.

        - **Feature Extraction**:

        **Spectrograms**: Visualize the frequency content of the audio signal over time. You can use libraries like Librosa or SciPy to compute spectrograms.
                
        **MFCCs**: These are commonly used features for speech recognition tasks. They capture the short-term power spectrum of a sound and are widely used in speaker recognition.
        - Other features: You can also extract features such as pitch, energy, formants, etc., depending on your specific requirements.
        
        **Feature Representation**: Once you have extracted the features, you'll typically represent them in a format suitable for machine learning algorithms. This could involve flattening the spectrogram or MFCC matrices into one-dimensional vectors, for example.
        
        **Speaker Identification Model**: Train a machine learning model or a deep learning model on your dataset. You can use techniques such as Support Vector Machines (SVMs), Random Forests, Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), or Transformer models.
        Use labeled data where each audio clip is associated with the identity of the speaker.
        
        **Evaluation**: Evaluate the performance of your model using metrics such as accuracy, precision, recall, or F1-score.
        Use techniques like cross-validation to ensure the generalization of your model.
                
        **Fine-tuning and Optimization**: Depending on the performance of your model, you might need to fine-tune hyperparameters, try different architectures, or explore data augmentation techniques to improve performance.
        
        **Inference**:
        Once your model is trained and validated, you can use it to predict speaker identities on unseen audio data.
        Remember, the success of your speaker identification system will depend on various factors such as the quality and diversity of your dataset, the choice of features, and the effectiveness of your machine learning model. Experimentation and iterative improvements are key to building a robust system.
    """)