import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# setup
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# loading the dataset
def load_dataset(file_path, tokenizer):
    dataset = TextDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        block_size=128 
    )
    return dataset

dataset_path = "gpttraining.txt"
train_dataset = load_dataset(dataset_path, tokenizer)

# training args
training_args = TrainingArguments(
    output_dir="./gptModel",
    overwrite_output_dir=True,
    num_train_epochs=100, # 100 training cycles
    per_device_train_batch_size=4,
    save_total_limit=1,
    prediction_loss_only=True,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    ),
)

trainer.train()

model.save_pretrained("gptModel")
tokenizer.save_pretrained("gptModel")
print("training complete.")
