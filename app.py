import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Movie Review Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment & Insight")
st.markdown("Enter a movie review below to analyze its sentiment and get a one-sentence summary.")

# ----- 加载模型 -----
@st.cache_resource
def load_sentiment_pipeline():
    """加载你微调的 IMDb 情感分析模型 (DistilBERT)"""
    return pipeline(
        "text-classification",
        model="Aku0423/imdb-distilbert",
        tokenizer="Aku0423/imdb-distilbert",
        return_all_scores=False,
    )

@st.cache_resource
def load_summarizer():
    """加载 BART 摘要模型 (适配新版 transformers)"""
    return pipeline("text-generation", model="facebook/bart-large-cnn")

sentiment_pipe = load_sentiment_pipeline()
summarizer_pipe = load_summarizer()

# ----- 用户输入 -----
review = st.text_area(
    "✍️ Paste a movie review (at least 20 words):",
    height=200,
    placeholder="e.g. 'This film is a masterpiece...'"
)

if st.button("🔍 Analyze", type="primary"):
    # 简单长度检查
    if len(review.split()) < 10:
        st.warning("Please enter a longer review (at least 10 words).")
    else:
        with st.spinner("Analyzing sentiment and generating summary..."):

            # 1. 情感分析（使用完整原文）
            sentiment = sentiment_pipe(review)[0]
            # 你的模型已自带 id2label，直接返回 POSITIVE / NEGATIVE
            label = f"{sentiment['label']} 😊" if sentiment['label'] == 'POSITIVE' else f"{sentiment['label']} 😞"
            confidence = sentiment['score']

            # 2. 摘要（截断后再生成）
            review_for_summary = review[:1024]  # BART 最大输入长度
            prompt = f"summarize: {review_for_summary}"
            raw_summary = summarizer_pipe(
                prompt,
                max_new_tokens=60,
                min_new_tokens=20,
                do_sample=False,
            )[0]['generated_text']

            # 如果生成结果包含了 prompt 本身，则移除
            if raw_summary.startswith(prompt):
                summary = raw_summary[len(prompt):].strip()
            else:
                summary = raw_summary.strip()

        # ----- 显示结果 -----
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment")
            st.metric(label=label, value=f"{confidence:.2%}")
        with col2:
            st.subheader("Key Insight")
            st.write(summary)

        st.divider()
        st.caption("Powered by your fine-tuned DistilBERT (trained on Stanford IMDb) and Facebook BART.")
