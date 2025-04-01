import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from models.moevar import MOEVAR
from datasets import load_dataset

def train_moevar(
    model: MOEVAR,
    train_dataset: DataLoader,
    val_dataset: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=100, num_training_steps=len(train_dataset) * num_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataset:
            labels, inputs = batch["labels"].to(device), batch["inputs"].to(device)
            logits = model(labels, inputs)
            loss = loss_fn(logits.view(-1, logits.size(-1)), inputs.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_dataset:
                labels, inputs = batch["labels"].to(device), batch["inputs"].to(device)
                logits = model(labels, inputs)
                val_loss += loss_fn(logits.view(-1, logits.size(-1)), inputs.view(-1)).item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss / len(val_dataset)}")
