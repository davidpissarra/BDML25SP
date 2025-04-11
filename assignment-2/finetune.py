import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import gc
from torch.utils.data import DataLoader


BASE_DIR = "/home/ddp8196/BDML25SP/"
MODEL_NAME = BASE_DIR + "Llama-3.2-3B"
TRAIN_DS_FOLDER = BASE_DIR + "dataset_txt_small/train/"
TEST_DS_FOLDER = BASE_DIR + "dataset_txt_small/test/"
DEVICE = "cuda:0"
IGNORE_INDEX = -100
ATTN_IGNORE_INDEX = 0

# Hyperparemeters
BATCH_SIZE = 2  # batch size
GA_STEPS = 8  # gradient acc. steps
EPOCHS = 10
LR = 1e-5

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# disable KV cache due to memory consumption (since not needed for training)
model.config.use_cache = False

# function to tokenize dataset
def tokenize(ds_element):
    tokenized_text = tokenizer(ds_element["text"], truncation=False, padding=False, add_special_tokens=False)
    return {
        "input_ids": tokenized_text["input_ids"],
        "labels": tokenized_text["input_ids"],
        "attention_mask": tokenized_text["attention_mask"],
    }

# function to batch inputs
def collate(elements):
    # Extract input_ids from each element and find the maximum length among them
    tokens = [e["input_ids"] for e in elements]
    tokens_maxlen = max([len(t) for t in tokens])

    for e in elements:
        input_ids = e["input_ids"]
        labels = e["labels"]
        attention_mask = e["attention_mask"]

        # Calculate the padding length required to match the maximum token length
        pad_len = tokens_maxlen - len(input_ids)

        # Pad 'input_ids' with the pad token ID, 'labels' with IGNORE_INDEX, and 'attention_mask' with 0
        input_ids.extend(pad_len * [tokenizer.pad_token_id])
        labels.extend(pad_len * [IGNORE_INDEX])
        attention_mask.extend(pad_len * [ATTN_IGNORE_INDEX])

    # create and return batch with all the data in elements
    batch = {
        "input_ids": torch.tensor([e["input_ids"] for e in elements]),
        "labels": torch.tensor([e["labels"] for e in elements]),
        "attention_mask": torch.tensor([e["attention_mask"] for e in elements]),
    }
    return batch

def train():
    # Load dataset
    dataset = load_dataset(
        "text",
        data_files={
            "train": [TRAIN_DS_FOLDER + filename for filename in os.listdir(TRAIN_DS_FOLDER)],
            "test": [TEST_DS_FOLDER + filename for filename in os.listdir(TEST_DS_FOLDER)],
        }
    )

    # apply tokenize
    dataset_tokenized = dataset.map(
        tokenize,
        batched=False,
        num_proc=1,
        remove_columns=["text"],
    )

    steps_per_epoch = len(dataset_tokenized["train"]) // (BATCH_SIZE * GA_STEPS)

    # Create DataLoader for the tokenized dataset
    train_dataloader = DataLoader(
        dataset_tokenized["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )

    # training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            if (step + 1) % GA_STEPS == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / steps_per_epoch}")
        torch.cuda.empty_cache()
        gc.collect()

    # Save model after all epochs
    output_dir = os.path.join(BASE_DIR, "lab2/checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, f"single_gpu_checkpoint_epoch_{EPOCHS}"))

if __name__ == "__main__":
    train()