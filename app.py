import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Movie Review Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment & Insight")
st.markdown("Enter a movie review below to analyze its sentiment and get a one-sentence summary.")

# ----- 情感分析 pipeline (你的微调模型) -----
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline(
        "text-classification",
        model="Aku0423/imdb-distilbert",
        tokenizer="Aku0423/imdb-distilbert"
    )

# ----- 摘要模型 (直接加载，不使用 text-generation pipeline) -----
@st.cache_resource
def load_summarizer_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

sentiment_pipe = load_sentiment_pipeline()
sum_tokenizer, sum_model = load_summarizer_model()

# ----- UI -----
review = st.text_area("✍️ Paste a movie review (at least 20 words):", height=200)

if st.button("🔍 Analyze", type="primary"):
    if len(review.split()) < 10:
        st.warning("Please enter a longer review (at least 10 words).")
    else:
        with st.spinner("Analyzing sentiment and generating summary..."):
            # 1. 情感分析
            sentiment = sentiment_pipe(review)[0]
            label = "POSITIVE 😊" if sentiment['label'] == 'POSITIVE' else "NEGATIVE 😞"
            conf = sentiment['score']

            # 2. 摘要
            review_trim = review[:1024]
            inputs = sum_tokenizer(review_trim, return_tensors="pt", truncation=True, max_length=1024)
            summary_ids = sum_model.generate(
                inputs.input_ids,
                max_length=60,
                min_length=20,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # 显示
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment")
            st.metric(label=label, value=f"{conf:.2%}")
        with col2:
            st.subheader("Key Insight")
            st.write(summary)

        st.divider()
        st.caption("Powered by your fine-tuned DistilBERT (trained on Stanford IMDb) and Facebook BART.")
