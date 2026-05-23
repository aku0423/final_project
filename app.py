import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Movie Review Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment & Insight")
st.markdown("Enter a movie review below to analyze its sentiment and get a one-sentence summary.")

# Pipeline 1: Your fine-tuned sentiment model (trusted source: Stanford IMDb + your fine-tuning)
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("text-classification", model="你的用户名/imdb-distilbert")

# Pipeline 2: Facebook BART summarizer (trusted source: Meta AI)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

sentiment_pipe = load_sentiment_pipeline()
summarizer = load_summarizer()

review = st.text_area("✍️ Paste a movie review (at least 20 words):", height=200)

if st.button("🔍 Analyze", type="primary"):
    if len(review.strip()) < 10:
        st.warning("Please enter a longer review.")
    else:
        with st.spinner("Analyzing sentiment and generating summary..."):
            sentiment = sentiment_pipe(review)[0]
            label = "POSITIVE 😊" if sentiment['label'] == 'POSITIVE' else "NEGATIVE 😞"
            confidence = sentiment['score']
            
            # Summarization (truncate if too long)
            if len(review) > 1024:
                review = review[:1024]
            summary = summarizer(review, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
            
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment")
            st.metric(label=label, value=f"{confidence:.2%}")
        with col2:
            st.subheader("Key Insight")
            st.write(summary)
        
        st.divider()
        st.caption("Powered by your fine-tuned DistilBERT (trained on Stanford IMDb) and Facebook BART.")
