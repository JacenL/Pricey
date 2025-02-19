import torch
from torch import nn
from transformers import Trainer, TrainingArguments, BertConfig, BertTokenizer
from datasets import load_from_disk
from model import PriceEstimator  # Import the modified BERT model

# try to use gpu, if not, cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
dataset = load_from_disk("C:/Users/PC/.vscode/Pricey/dataset")

# initializing bert
config = BertConfig.from_pretrained("bert-base-uncased")
model = PriceEstimator(config).to(device)

training_args = TrainingArguments(
    output_dir="./bertPriceModel",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
)

print("training started")
trainer.train()

outputDir = "bertPriceEstimator"
model.save_pretrained(outputDir)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained(outputDir)

print("Completed training and saved model and tokenizer")
