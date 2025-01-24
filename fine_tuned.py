from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import evaluate
import os

df = pd.read_csv(r"shuffled_DB.csv")

label_mapping = {"CREATE": 0, "READ": 1, "UPDATE": 2, "DELETE": 3, "DELETE_CONDITIONED_BASED": 4,
                 "READ_CONDITION_BASED_DATA": 5, "INSERT": 6}
df["labels"] = df["labels"].map(label_mapping)


train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["labels"].tolist(),
    test_size=0.3,
    random_state=42
)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "labels": test_labels})

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=7)

# Move model to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set dataset format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Verify PyTorch installation
try:
    print("PyTorch is available. Version:", torch.__version__)
except ImportError:
    print("PyTorch is not installed.")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",            # Directory to save checkpoints
    eval_strategy="epoch",      # Evaluate after every epoch
    save_strategy="epoch",            # Save after every epoch
    learning_rate=2e-5,               # Fine-tuning learning rate
    per_device_train_batch_size=20,   # Batch size for training
    per_device_eval_batch_size=16,    # Batch size for evaluation
    num_train_epochs=4,               # Number of epochs
    weight_decay=0.01,                # Weight decay
    logging_dir="./logs",             # Logging directory
    load_best_model_at_end=True,      # Load best model at the end
    metric_for_best_model="accuracy", # Metric to track the best model
    logging_steps=10                  # Log every 10 steps
)

# Load accuracy metric using evaluate
metric = evaluate.load("accuracy")

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Check if model is already saved
save_directory = "./final_model"
if not os.path.exists(save_directory):
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved in '{save_directory}'")
else:
    print(f"Model already exists in '{save_directory}'")

# Load the saved model and tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained(save_directory)
loaded_model = AutoModelForSequenceClassification.from_pretrained(save_directory).to(device)

# Predict function for new text input using the saved model
def predict_text(text):
    inputs = loaded_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the correct device
    loaded_model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        outputs = loaded_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return {v: k for k, v in label_mapping.items()}[predicted_class]

# Test the model on a sample text
sample_text = "Change the name of the entry which have age>0 and height<120 and whose count is 100"
predicted_intent = predict_text(sample_text)
print("Predicted Intent:", predicted_intent)

# # Evaluate the model on the test dataset
# def evaluate_model(test_dataset):
#     # Ensure the test dataset is formatted for PyTorch
#     test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

#     # Create DataLoader for test dataset
#     from torch.utils.data import DataLoader
#     from tqdm import tqdm

#     test_loader = DataLoader(test_dataset, batch_size=16)
#     correct = 0
#     total = 0
#     all_predictions = []
#     all_labels = []

#     loaded_model.eval()  # Set the model to evaluation mode
#     with torch.no_grad():  # Disable gradient calculations for evaluation
#         for batch in tqdm(test_loader, desc="Evaluating"):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             outputs = loaded_model(input_ids=input_ids, attention_mask=attention_mask)
#             logits = outputs.logits
#             predictions = torch.argmax(logits, dim=1)

#             # Update accuracy metrics
#             correct += (predictions == labels).sum().item()
#             total += labels.size(0)
            
#             # Collect predictions and labels for other metrics
#             all_predictions.extend(predictions.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     # Compute accuracy
#     accuracy = correct / total
#     print(f"Accuracy: {accuracy:.4f}")

#     # Use sklearn for additional metrics
#     from sklearn.metrics import classification_report, confusion_matrix

#     print("\nClassification Report:")
#     print(classification_report(all_labels, all_predictions, target_names=label_mapping.keys()))

#     print("\nConfusion Matrix:")
#     print(confusion_matrix(all_labels, all_predictions))

#     return accuracy

# # Call the evaluation function
# print("Evaluating the saved model...")
# accuracy = evaluate_model(test_dataset)
