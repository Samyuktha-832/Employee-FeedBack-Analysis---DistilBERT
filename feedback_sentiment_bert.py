import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import torch.nn.functional as F
import os

model_path = "saved_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

sentiment_labels = {0: "Positive", 1: "Negative", 2: "Neutral"}

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    sentiment = sentiment_labels[predicted_class.item()]
    confidence_score = int(confidence.item() * 100)
    return sentiment, confidence_score

def calculate_sentiment_summary(output_csv, summary_csv):
    df = pd.read_csv(output_csv)
    sentiment_counts = df["Predicted Sentiment"].value_counts(normalize=True) * 100
    sentiment_summary = pd.DataFrame({
        "Sentiment": sentiment_counts.index, 
        "Percentage (%)": sentiment_counts.values
    })
    sentiment_summary.to_csv(summary_csv, index=False)
    print("Sentiment summary saved to {summary_csv}")

bulk_feedback_csv = "feedback_predictions.csv"
manual_feedback_csv = "manual_feedback.csv"
summary_csv = "sentiment_summary.csv"

user_input = input("Do you want to enter feedback manually? (yes/no): ").strip().lower()

if user_input == "yes":
    while True:
        feedback_text = input("Enter your feedback (or type 'exit' to stop): ").strip()
        if feedback_text.lower() == "exit":
            break

        sentiment, confidence = predict_sentiment(feedback_text)

        
        df_feedback = pd.DataFrame([[feedback_text, sentiment, confidence]], 
                                   columns=["Feedback", "Predicted Sentiment", "Confidence (%)"])
        
        if os.path.exists(manual_feedback_csv):
            df_existing = pd.read_csv(manual_feedback_csv)
            df_feedback = pd.concat([df_existing, df_feedback], ignore_index=True)

        df_feedback.to_csv(manual_feedback_csv, index=False)
        
        print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence}%)")
        print(f"Feedback saved to {manual_feedback_csv}")

else:
    input_csv = "feedback_input.csv"

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file '{input_csv}' not found.")

    df = pd.read_csv(input_csv)

    if "feedback" not in df.columns:
        raise ValueError("The CSV file must contain a 'feedback' column.")

    df["Predicted Sentiment"], df["Confidence (%)"] = zip(*df["feedback"].apply(predict_sentiment))

    df.to_csv(bulk_feedback_csv, index=False)

    calculate_sentiment_summary(bulk_feedback_csv, summary_csv)

    print(f"Predictions saved to {bulk_feedback_csv}")






