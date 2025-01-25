from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__)

# Check if model is already saved
save_directory = "./final_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(save_directory):
    # Load the saved model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    model = AutoModelForSequenceClassification.from_pretrained(save_directory).to(device)
    print(f"Model and tokenizer loaded from '{save_directory}'")
else:
    print(f"Model not found in '{save_directory}', please ensure the model is saved before starting the server.")

# Predict function for new text input using the saved model
def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the correct device
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    # Reverse the label mapping to get the string corresponding to the class
    label_mapping = {
        0: "CREATE", 1: "READ", 2: "UPDATE", 3: "DELETE", 4: "DELETE_CONDITIONED_BASED",
        5: "READ_CONDITION_BASED_DATA", 6: "INSERT"
    }
    return label_mapping.get(predicted_class, "Unknown")

@app.route('/getIntent', methods=['POST'])
def post_data():
    data = request.get_json()
    if not data or 'paragraph' not in data:
        return jsonify({"response": "false"})
    
    intent = predict_text(data['paragraph'])
    print(data['paragraph'])
    
    return jsonify({"intent": intent})

if __name__ == '__main__':
    app.run(host='0.0.0.0' , debug=True)
