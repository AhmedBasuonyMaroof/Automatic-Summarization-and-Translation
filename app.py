import gc
import streamlit as st
from transformers import pipeline

# Define functions to load models only when needed
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def get_translator_en_to_ar():
    return pipeline("translation_en_to_ar", model="Helsinki-NLP/opus-mt-en-ar")

def get_translator_en_to_fr():
    return pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

# App title
st.title("Automatic Summarization and Translation")

# Instructions for the user
st.write("Enter a long text in the box below to get a summary and translation.")

# User input for the long text
user_input = st.text_area("Enter the text to be summarized:")

# Select the target language for translation
language = st.selectbox("Choose the target language for translation:", ["Arabic", "French"])

# Button to trigger summarization and translation
if st.button("Summarize Text"):
    if user_input.strip():
        with st.spinner("Processing..."):
            try:
                # Load models only when needed
                summarizer = get_summarizer()
                translator = get_translator_en_to_ar() if language == "Arabic" else get_translator_en_to_fr()

                # Summarize the input text
                summary = summarizer(user_input, max_length=150, min_length=30, do_sample=False)
                summarized_text = summary[0]['summary_text']
                st.subheader("Summary:")
                st.write(summarized_text)

                # Translate based on the chosen language
                translation = translator(summarized_text)
                st.subheader(f"Translation to {language}:")
                st.write(translation[0]['translation_text'])

            except Exception as e:
                st.error(f"An error occurred while processing the text: {e}")
                st.write(e)  # Log the error details for debugging
    else:
        st.warning("Please enter some text to summarize.")

    # Call garbage collection at the end to free up memory
    gc.collect()