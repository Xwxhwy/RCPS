import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Any

from .model import LPG_Transformer
from .tokenizer import LDLTokenizer


class LPG_Dataset(Dataset):
    """Loads (concept, ldl) pairs from a JSON Lines file."""

    def __init__(self, jsonl_path: str, tokenizer: LDLTokenizer, concept_preprocessor):
        self.data = [json.loads(line) for line in open(jsonl_path, 'r', encoding='utf-8')]
        self.tokenizer = tokenizer
        self.concept_preprocessor = concept_preprocessor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        concept_features = self.concept_preprocessor(item['concept'])
        tokenized_ldl = self.tokenizer.encode(item['ldl'])
        return {
            "concept_features": torch.tensor(concept_features, dtype=torch.float32),
            "ldl_tokens": torch.tensor(tokenized_ldl, dtype=torch.long)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], padding_value: int) -> Dict[str, torch.Tensor]:
    """Pads sequences to the maximum length in a batch."""
    concept_features = torch.stack([item['concept_features'] for item in batch])
    ldl_tokens = [item['ldl_tokens'] for item in batch]
    padded_ldl_tokens = nn.utils.rnn.pad_sequence(ldl_tokens, batch_first=True, padding_value=padding_value)
    return {"concept_features": concept_features, "ldl_tokens": padded_ldl_tokens}


class ConceptPreprocessor:
    """A mock preprocessor to convert JSON concept objects into fixed-size vectors for demonstration."""

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim

    def __call__(self, concept: Dict[str, Any]) -> List[float]:
        # This is a simplified featurization. A production system would use real encoders.
        vector = [
            concept.get("num_text_elements", 0) / 10.0,
            concept.get("num_image_elements", 0) / 5.0,
            concept.get("total_text_length", 0) / 1000.0,
        ]
        padding_size = self.feature_dim - len(vector)
        vector.extend([0.0] * padding_size)
        return vector[:self.feature_dim]


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_value):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        concept_features = batch['concept_features'].to(device)
        ldl_tokens = batch['ldl_tokens'].to(device)

        inputs = ldl_tokens[:, :-1]
        targets = ldl_tokens[:, 1:]

        logits = model(concept_features, inputs)
        loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            concept_features = batch['concept_features'].to(device)
            ldl_tokens = batch['ldl_tokens'].to(device)

            inputs = ldl_tokens[:, :-1]
            targets = ldl_tokens[:, 1:]

            logits = model(concept_features, inputs)
            loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
            total_loss += loss.item()

    return total_loss / len(loader)


def train_lpg_model(config: Dict):
    """
    Main training function for the Layout Prototype Generator model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = LDLTokenizer(vocab_path=config['vocab_path'])
    concept_preprocessor = ConceptPreprocessor(feature_dim=config['model']['input_dim'])

    train_dataset = LPG_Dataset(config['train_dataset_path'], tokenizer, concept_preprocessor)
    val_dataset = LPG_Dataset(config['val_dataset_path'], tokenizer, concept_preprocessor)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )

    model = LPG_Transformer(
        input_dim=config['model']['input_dim'],
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_layers'],
        num_decoder_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_ff'],
        padding_idx=tokenizer.pad_token_id
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    print("Starting training...")
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config['grad_clip'])
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{config['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Perplexity: {torch.exp(torch.tensor(val_loss)):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"Model saved to {config['model_save_path']} at epoch {epoch + 1}")

    print("Training complete.")


