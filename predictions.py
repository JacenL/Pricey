from transformers import BertTokenizer, BertConfig
import torch
from safetensors.torch import load_file
from model import PriceEstimator
import joblib

# tokenizer
tokenizer = BertTokenizer.from_pretrained("bertPriceEstimator")

config = BertConfig.from_pretrained("bertPriceEstimator")

# load model 
model = PriceEstimator(config)
model.load_state_dict(load_file("bertPriceEstimator/model.safetensors", device="cpu"))
model.eval()

# predict price
def predictPrice(description):
    inputs = tokenizer(description, return_tensors="pt", padding="max_length", truncation=True)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        predictedPrice = model(**inputs).item()
    return round(predictedPrice, 2)

# example
description = "HP Wireless Silent 280M Mouse"
predictedPrice = predictPrice(description)
print(f"Predicted Price: ${predictedPrice}")
