import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = "gptModel"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def predict_price(product_description):
    prompt = f"The price of {product_description} is $"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    # generate price prediction
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            do_sample = True,
            pad_token_id=tokenizer.eos_token_id
        )

    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
    first_line = prediction.split("\n")[0]

    return first_line

"""descriptions = [
    "an office mouse",
    "a gaming laptop with an RTX 4090",
    "an Apple iPhone 14 Pro Max",
    "a wireless Bluetooth headset",
    "a 55-inch 4K Smart TV"
]"""

while True:
    description = input("Enter a product description (type 'exit' to quit): ")
    if description.lower() == "exit":
        break
    predicted_price = predict_price(description)
    print(predicted_price)
