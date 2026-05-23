import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

# Download NLTK sentence tokenizer (only needed once, cached after first run)
@st.cache_resource
def download_nltk_punkt():
    nltk.download('punkt_tab', quiet=True)
    return True

# ----- Load models (cached) -----
@st.cache_resource
def load_sentiment_pipeline():
    """Load your fine-tuned IMDb sentiment classifier."""
    return pipeline(
        "text-classification",
        model="Aku0423/imdb-distilbert",
        tokenizer="Aku0423/imdb-distilbert",
        truncation=True,
        max_length=512,
    )

@st.cache_resource
def load_summarizer_model():
    """Load BART-large-CNN for extractive/abstractive summarization."""
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Initialize
download_nltk_punkt()
sentiment_pipe = load_sentiment_pipeline()
sum_tokenizer, sum_model = load_summarizer_model()

# ----- Streamlit UI -----
st.set_page_config(page_title="Movie Review Analyzer", page_icon="🎬")
st.title("🎬 Movie Review Sentiment & Insight")
st.markdown("Enter a movie review below to analyze its sentiment and get a **concise insight** (1–2 sentences).")

review = st.text_area(
    "✍️ Paste a movie review (at least 10 words):",
    height=200,
    placeholder="e.g. 'The plot was boring and predictable, the acting felt wooden...'"
)

if st.button("🔍 Analyze", type="primary"):
    if len(review.split()) < 10:
        st.warning("Please enter a longer review (at least 10 words).")
    else:
        with st.spinner("Analyzing sentiment and generating insight..."):
            # 1. Sentiment analysis (on full text)
            sentiment_result = sentiment_pipe(review)[0]
            label = sentiment_result['label']  # POSITIVE / NEGATIVE
            confidence = sentiment_result['score']
            display_label = f"{label} 😊" if label == "POSITIVE" else f"{label} 😞"

            # 2. Summarization (truncate to avoid memory issues)
            review_trim = review[:1024]
            inputs = sum_tokenizer(
                review_trim,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            summary_ids = sum_model.generate(
                inputs.input_ids,
                max_length=80,
                min_length=20,
                num_beams=6,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            raw_summary = sum_tokenizer.decode(
                summary_ids[0], skip_special_tokens=True
            )

            # Post-process: ensure complete sentence, limit to 2 sentences
            if raw_summary and not raw_summary.endswith('.'):
                raw_summary += '.'
            sentences = nltk.sent_tokenize(raw_summary)
            if len(sentences) > 2:
                raw_summary = ' '.join(sentences[:2])

        # ----- Display results -----
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment")
            st.metric(label=display_label, value=f"{confidence:.2%}")
        with col2:
            st.subheader("Key Insight")
            st.write(raw_summary)

        st.divider()
        st.caption(
            "Powered by your fine-tuned DistilBERT (trained on Stanford IMDb) "
            "and Facebook BART-large-CNN for summarization."
        )
