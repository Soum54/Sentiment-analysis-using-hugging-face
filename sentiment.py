import streamlit as st
import pandas as pd
from transformers import pipeline

# Load the sentiment analysis pipeline
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Define a function to analyze sentiment and return emoji and score
def analyze_sentiment(text):
    result = pipe(text)[0]  # Get the first (and only) result
    sentiment = result['label'].strip().upper()  # Normalize sentiment label
    score = result['score']  # Get the confidence score

    # Map the sentiment to an emoji
    if sentiment == "POSITIVE":
        emoji = "😊"
    elif sentiment == "NEGATIVE":
        emoji = "😢"
    elif sentiment == "NEUTRAL":
        emoji = "😐"
    else:
        emoji = "🤔"  # Fallback for any other cases

    # Format the score as a percentage with two decimal places
    score_percent = f"{score * 100:.2f}%"

    return sentiment.capitalize(), emoji, score_percent

# Streamlit UI elements
st.title("Sentiment Analysis using Hugging Face")

st.write("""
Upload a CSV file with a 'Text' column or input your own text to perform sentiment analysis.
The model used is fine-tuned for customer feedback sentiment analysis.
""")

# User text input for single sentiment analysis
st.header("Analyze Sentiment of a Single Text")
user_input = st.text_input("Enter text for sentiment analysis")

# Add analyze button for single text input
if st.button("Analyze Text"):
    if user_input:
        sentiment, emoji, score = analyze_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment} {emoji}")
        st.write(f"**Confidence Score:** {score}")
    else:
        st.warning("Please enter some text to analyze.")

# File upload for batch sentiment analysis
st.header("Batch Sentiment Analysis via CSV")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Add analyze button for CSV file
if uploaded_file:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Check if the 'Text' column exists
    if 'Text' in df.columns:
        st.write("**Data Preview:**")
        st.write(df.head())

        if st.button("Analyze CSV"):
            # Perform sentiment analysis and store results
            st.write("Performing sentiment analysis...")
            sentiments = df['Text'].apply(analyze_sentiment)

            # Split the sentiments into separate columns
            df['Sentiment'] = sentiments.apply(lambda x: x[0])
            df['Emoji'] = sentiments.apply(lambda x: x[1])
            df['Confidence Score'] = sentiments.apply(lambda x: x[2])

            # Show the result in Streamlit
            st.write("**Sentiment analysis results:**")
            st.write(df.head())

            # Provide a download button for the updated CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name='sentiment_analysis_results_with_emoji_and_score.csv',
                mime='text/csv',
            )
    else:
        st.error("The uploaded CSV does not contain a 'Text' column.")
else:
    st.write("Please upload a CSV file to perform batch sentiment analysis.")
