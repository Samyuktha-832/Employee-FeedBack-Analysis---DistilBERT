from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
from feedback_sentiment_bert import predict_sentiment, calculate_sentiment_summary

app = Flask(__name__)

# Homepage Route
@app.route('/')
def homepage():
    return render_template('homepage.html')

# Feedback Page Route
@app.route('/feedback')
def feedback_page():
    return render_template('feedback.html')

# Dashboard Route
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# API for Manual Feedback Sentiment Prediction
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    feedback = data.get("feedback", "").strip()

    if not feedback:
        return jsonify({"error": "Feedback is empty"}), 400

    sentiment, confidence = predict_sentiment(feedback)

    # Save the feedback
    feedback_data = pd.DataFrame([[feedback, sentiment, confidence]], 
                                 columns=["Feedback", "Predicted Sentiment", "Confidence (%)"])
    
    manual_feedback_csv = "manual_feedback.csv"
    if os.path.exists(manual_feedback_csv):
        existing_data = pd.read_csv(manual_feedback_csv)
        feedback_data = pd.concat([existing_data, feedback_data], ignore_index=True)

    feedback_data.to_csv(manual_feedback_csv, index=False)

    return jsonify({"sentiment": sentiment, "confidence": confidence})

# API for Bulk CSV Sentiment Analysis
@app.route('/api/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = "feedback_input.csv"
    file.save(file_path)

    # Read CSV and Predict Sentiment
    df = pd.read_csv(file_path)

    if "feedback" not in df.columns:
        return jsonify({"error": "CSV must contain a 'feedback' column"}), 400

    df["Predicted Sentiment"], df["Confidence (%)"] = zip(*df["feedback"].apply(predict_sentiment))
    df.to_csv("feedback_predictions.csv", index=False)

    # Generate Summary
    summary_csv = "sentiment_summary.csv"
    calculate_sentiment_summary("feedback_predictions.csv", summary_csv)

    return jsonify({"summaryFile": summary_csv})

# API to Download Sentiment Summary
@app.route('/download/summary')
def download_summary():
    return send_file("sentiment_summary.csv", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
