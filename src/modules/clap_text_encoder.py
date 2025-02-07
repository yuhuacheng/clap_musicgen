import hashlib
from typing import List, Optional

import torch
import torch.nn as nn

from transformers import RobertaModel, PreTrainedModel, PretrainedConfig
from src.modules.mlp_projection import MLPProjection


class ClapTextEncoderConfig(PretrainedConfig):
    model_type = "clap_text_encoder"

    def __init__(self, projection_dim: int = None, hidden_dim: int = None, freeze_text_encoder: bool = False, **kwargs):
        super().__init__(**kwargs)        
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.freeze_text_encoder = freeze_text_encoder


class ClapTextEncoder(PreTrainedModel):
    """
    A wrapper around a HuggingFace text encoder that adds trainable MLP projection layers.

    Attributes:
        text_encoder: The pretrained text encoder model.
        projection: An MLP projection head that maps embeddings to a shared space.
        freeze_text_encoder: A flag to freeze text encoder weights during training.
    """
    config_class = ClapTextEncoderConfig

    def __init__(self, config: ClapTextEncoderConfig):
        super().__init__(config)

        # Load the pretrained text encoder
        # note that the pooler layers are randomly initiated since it wasn't trained with any classification task head
        self.text_encoder = RobertaModel.from_pretrained("model/roberta_finetuned")
        
        # Add a projection head on top of the last hidden layer
        input_dim = self.text_encoder.pooler.dense.in_features
        self.projection = MLPProjection(input_dim=input_dim, output_dim=config.projection_dim, hidden_dim=config.hidden_dim)        

        # Handle freezing of text encoder weights except for pooler
        self.freeze_text_encoder = config.freeze_text_encoder
        if self.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # TODO: check if I still need this pooler weights freezing
        for param in self.text_encoder.pooler.parameters():
            param.requires_grad = True

        # Cache for storing computed embeddings
        self.embedding_cache = {}

    def forward(self, ids: Optional[List[str]], input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        x = self.get_pretrained_text_emb(ids, input_ids, attention_mask, **kwargs)
        return self.projection(x)  # Shape: (B, projection_dim)
    
    def get_pretrained_text_emb(self, ids: Optional[List[str]], input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        use_cache = self.freeze_text_encoder and ids is not None

        # Check if all IDs in the batch are cached
        if use_cache and all(id in self.embedding_cache for id in ids):
            return torch.stack([self.embedding_cache[id] for id in ids])  # Shape: (B, hidden_dim)

        # Compute embeddings for the full batch
        with torch.no_grad() if self.freeze_text_encoder else torch.enable_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            hidden_states = outputs.last_hidden_state
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())  # Shape: (B, seq_length, hidden_dim)
                sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
                sum_mask = attention_mask.sum(dim=1)
            else:
                sum_embeddings = torch.sum(hidden_states, dim=1)
                sum_mask = hidden_states.shape[1]  # seq_length

            sentence_embeddings = sum_embeddings / sum_mask  # Element-wise division

        # Cache each sample separately only if IDs are provided
        if use_cache:
            for i, sample_id in enumerate(ids):
                self.embedding_cache[sample_id] = sentence_embeddings[i]

        return sentence_embeddings  # Shape: (B, hidden_dim)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pretrained model and ensure that pooler weights are included in training.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        if model.config.freeze_text_encoder:
            for param in model.text_encoder.parameters():
                param.requires_grad = False

        # Always ensure pooler weights are trainable
        for param in model.text_encoder.pooler.parameters():
            param.requires_grad = True
        return model
