import streamlit as st
from transformers import pipeline

# ----- Load two pipelines (cached) -----
@st.cache_resource
def load_sentiment_pipeline():
    """Pipeline 1: Fine-tuned IMDb sentiment classifier."""
    return pipeline(
        "text-classification",
        model="Aku0423/imdb-distilbert",
        tokenizer="Aku0423/imdb-distilbert",
        truncation=True,
        max_length=512,
    )

@st.cache_resource
def load_summarization_pipeline():
    """Pipeline 2: BART summarizer using text-generation pipeline."""
    return pipeline(
        "text-generation",
        model="facebook/bart-large-cnn",
    )

sentiment_pipe = load_sentiment_pipeline()
summarizer = load_summarization_pipeline()

# ----- Helper: limit output to first two sentences -----
def first_two_sentences(text):
    """Return the first two sentences from a text (split on . ! ?)."""
    sentences = []
    current = ""
    for ch in text:
        current += ch
        if ch in ('.', '!', '?'):
            stripped = current.strip()
            if stripped:
                sentences.append(stripped)
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return ' '.join(sentences[:2])

# ----- UI -----
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
            # --- Sentiment analysis ---
            sentiment_result = sentiment_pipe(review)[0]
            label = sentiment_result['label']          # POSITIVE or NEGATIVE
            confidence = sentiment_result['score']
            display_label = f"{label} 😊" if label == "POSITIVE" else f"{label} 😞"

            # --- Summarization ---
            review_trim = review[:1024]   # BART max input (truncate if needed)
            prompt = f"summarize: {review_trim}"

            raw_output = summarizer(
                prompt,
                max_new_tokens=80,
                min_new_tokens=20,
                do_sample=False,
                num_beams=6,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )[0]['generated_text']

            # Clean possible repetition of the prompt
            if raw_output.startswith(prompt):
                raw_output = raw_output[len(prompt):].strip()

            # Ensure it ends with a proper punctuation
            if raw_output and raw_output[-1] not in ('.', '!', '?'):
                raw_output += '.'

            insight = first_two_sentences(raw_output)

        # --- Display results ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment")
            st.metric(label=display_label, value=f"{confidence:.2%}")
        with col2:
            st.subheader("Key Insight")
            st.write(insight)

        st.divider()
        st.caption(
            "Powered by your fine-tuned DistilBERT (trained on Stanford IMDb) "
            "and Facebook BART-large-CNN for summarization."
        )
