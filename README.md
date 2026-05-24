# Fine-tuned Model Information
Our fine-tuned sentiment analysis model is publicly available on Hugging Face Hub:
https://huggingface.co/Aku0423/imdb-distilbert

## Model Details
- Base Model: distilbert-base-uncased
- Task: Binary sentiment classification (POSITIVE/NEGATIVE)
- Accuracy: 90.15% on 2,000 balanced test samples
- Model Size: 268MB
- Training Time: 8.5 minutes

## How to Use
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="Aku0423/imdb-distilbert")
result = classifier("This movie was amazing!")
print(result)
