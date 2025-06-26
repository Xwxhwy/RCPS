# training/lpg_training/model.py

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Injects positional information into sequence embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Tensor, shape [seq_len, batch_size, d_model]"""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LPG_Transformer(nn.Module):
    """
    Layout Prototype Generator (LPG) based on a Transformer architecture.
    Maps a slide concept feature vector to a symbolic Layout Description Language (LDL) sequence.
    """

    def __init__(self,
                 input_dim: int,
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 padding_idx: int = 0):
        """
        Initializes the LPG Transformer model.

        Args:
            input_dim: Dimension of the input slide concept features.
            vocab_size: Size of the LDL vocabulary.
            d_model: The number of expected features in the encoder/decoder inputs.
            nhead: The number of heads in the multiheadattention models.
            num_encoder_layers: The number of sub-encoder-layers in the encoder.
            num_decoder_layers: The number of sub-decoder-layers in the decoder.
            dim_feedforward: The dimension of the feedforward network model.
            dropout: The dropout value.
            activation: The activation function of encoder/decoder intermediate layer.
            padding_idx: Index of the padding token in the vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx

        self.input_projection = nn.Linear(input_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.padding_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
            batch_first=True
        )
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src_features: torch.Tensor, tgt_sequence: torch.Tensor) -> torch.Tensor:
        """
        Defines the training-time forward pass using teacher forcing.

        Args:
            src_features: Slide concept features, shape [batch_size, input_dim].
            tgt_sequence: Ground truth LDL sequence, shape [batch_size, seq_len].

        Returns:
            Output logits, shape [batch_size, seq_len, vocab_size].
        """
        # unsqueeze(1) to treat the single feature vector as a sequence of length 1
        src_encoded = self.input_projection(src_features).unsqueeze(1)

        tgt_padding_mask = (tgt_sequence == self.padding_idx)
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt_sequence.size(1), device=src_features.device
        )

        tgt_emb = self.embedding(tgt_sequence) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)

        output = self.transformer(
            src=src_encoded,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        return self.generator(output)

    @torch.no_grad()
    def generate(self, src_features: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int = 128) -> torch.Tensor:
        """
        Generates an LDL sequence given concept features using greedy decoding.

        Args:
            src_features: Slide concept features, shape [batch_size, input_dim].
            sos_idx: Start-of-sequence token ID.
            eos_idx: End-of-sequence token ID.
            max_len: Maximum length of the generated sequence.

        Returns:
            Generated LDL token sequence, shape [batch_size, seq_len].
        """
        self.eval()
        batch_size = src_features.size(0)

        src_encoded = self.input_projection(src_features).unsqueeze(1)
        memory = self.transformer.encoder(src_encoded)

        generated_seq = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src_features.device)

        for _ in range(max_len - 1):
            tgt_emb = self.embedding(generated_seq) * math.sqrt(self.d_model)
            tgt_emb = self.positional_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)

            output = self.transformer.decoder(tgt=tgt_emb, memory=memory)
            logits = self.generator(output[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated_seq = torch.cat([generated_seq, next_token], dim=1)

            if (next_token.squeeze() == eos_idx).all():
                break

        return generated_seq