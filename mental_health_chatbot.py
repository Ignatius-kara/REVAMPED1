import json
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer

# Load the dataset to map predictions to responses
with open("fine_tune_mental_health_5000_dataset.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
df = pd.DataFrame(data)

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained("./mental_health_model")
tokenizer = BertTokenizer.from_pretrained("./mental_health_model")

# Load the label encoder
mlb_classes = np.load("label_encoder.npy", allow_pickle=True)
mlb = MultiLabelBinarizer()
mlb.fit([mlb_classes])

# Function to predict labels for a user message
def predict_labels(user_message):
    encoding = tokenizer(
        user_message,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        preds = torch.sigmoid(logits).numpy()

    # Apply threshold to get multi-label predictions
    threshold = 0.5
    predicted_labels = (preds > threshold).astype(int)
    labels = mlb.inverse_transform(predicted_labels)[0]
    return labels

# Function to map predicted labels to a chatbot response
def get_response(user_message, predicted_labels):
    intent, emotion, suggested_action = predicted_labels

    # Filter dataset for similar intent, emotion, and suggested action
    filtered_df = df[
        (df["intent"] == intent) &
        (df["emotion"] == emotion) &
        (df["suggested_action"] == suggested_action)
    ]

    # If a match is found, return the most common response
    if not filtered_df.empty:
        return filtered_df["chatbot_response"].mode()[0]
    else:
        # Fallback response if no match is found
        return "I'm here to help. Can you tell me more about how you're feeling?"

# Main function to process user input
def process_user_input(user_message):
    predicted_labels = predict_labels(user_message)
    response = get_response(user_message, predicted_labels)
    return response

# Example usage
if __name__ == "__main__":
    user_input = "I feel like I'm not good enough."
    response = process_user_input(user_input)
    print(f"User: {user_input}")
    print(f"Chatbot: {response}")
