import pandas as pd
import torch
from datasets import Dataset
from transformers import BertTokenizer
from sklearn.preprocessing import MinMaxScaler

file_path = "C:/Users/PC/.vscode/Pricey/amazonprices.csv"
df = pd.read_csv(file_path)

# make prices numeric if they aren't
df["price"] = pd.to_numeric(df["price"], errors="coerce")

scaler = MinMaxScaler()
df["price"] = scaler.fit_transform(df[["price"]].values.reshape(-1, 1))

df["item"] = df["title"] + " " + df["categories"]

# convert data to HuggingFace dataset format
dataset = Dataset.from_pandas(df.reset_index(drop=True))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# tokenize product descriptions
def tokenize_function(examples):
    return tokenizer(examples["item"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

datasetPath = "C:/Users/PC/.vscode/Pricey/dataset"
tokenized_datasets.save_to_disk(datasetPath)

import joblib
joblib.dump(scaler, "bertPriceEstimator/price_scaler.pkl")


print(f"dataset creation completed")
