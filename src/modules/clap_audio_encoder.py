import hashlib
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.musicgen.modeling_musicgen import MusicgenDecoder
from transformers.models.encodec.modeling_encodec import EncodecModel

from src.modules.mlp_projection import MLPProjection


class MusicGenEmbedder:
    """
    The main class for extracting the embedding through MusicGen model given an audio waveform. The extraction steps
    are as follows,
        1. Pass original audio waveform to EnCodec Encoder, to get discrete audio tokens
        2. Pass the audio tokens to MusicGen Decoder (Decoder LM), to predict the next audio token
        3. Extract the last hidden layer (before the softmax output layer) from the transformer, and take only the
        last token’s hidden state from the output sequence as the final audio embedding, since attention ensures the
        last token is attended by all previous token
    """
    def __init__(self,
                 encodec: EncodecModel,
                 transformer: MusicgenDecoder
                 ):
        self.encodec = encodec
        self.transformer = transformer
    
    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        The main entry function to get the embedding from a waveform

        :param waveform: tensor with shape of (C, T), where C: num channels, T: num samples
        :return: embedding with shape (B, E), where B: number of segments, E: embedding size
        """
        _, C, _ = waveform.shape
        assert C == 1, "currently only support mono waveform"
        
        tokens = self._tokenize_with_encodec(waveform)  # (B, K, T')
        emb = self._embed_with_transformer(tokens)  # (B, E)
        return emb

    def _tokenize_with_encodec(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Get discrete tokens for each segment from the EnCodec model's encoder
        """        
        output = self.encodec.encode(waveform)
        tokens = output.audio_codes
        return tokens

    def _embed_with_transformer(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward the EnCodec tokens to MusicGen LM's transfomer decoder, and retrieve its hidden states as embeddings.
        """
        tokens = tokens.squeeze(0)
        _, K, _ = tokens.shape

        x = sum([self.transformer.embed_tokens[codebook](tokens[:, codebook]) for codebook in range(K)])
        out = self.transformer(inputs_embeds=x)
        lhs = out.last_hidden_state

        # take only the hidden state from the last token in the input sequence, since the last token is attended by all the previous tokens in the transformer
        B, _, E = lhs.shape
        emb = lhs[:, -1, :].view(B, E)
        return emb


class ClapMusicGenAudioEncoderConfig(PretrainedConfig):
    model_type = "clap_musicgen_audio_encoder"

    def __init__(self, encodec_model_name: str = None, musicgen_decoder_model_name: str = None, projection_dim: int = None, hidden_dim: int = None, freeze_musicgen: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.encodec_model_name = encodec_model_name
        self.musicgen_decoder_model_name = musicgen_decoder_model_name
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.freeze_musicgen = freeze_musicgen


class ClapMusicGenAudioEncoder(PreTrainedModel):
    """
    A wrapper around the MusicGen audio encoder that adds trainable MLP projection layers.

    Attributes:
        encodec: The EnCodec model used for tokenizing audio inputs.
        transformer: The pretrained MusicGen decoder model.
        projection: An MLP projection head that maps embeddings to a shared space.
        freeze_musicgen: A flag to freeze MusicGen weights during training.
    """
    config_class = ClapMusicGenAudioEncoderConfig

    def __init__(self, config: ClapMusicGenAudioEncoderConfig):
        super().__init__(config)

        # Load the EnCodec and MusicGen decoder        
        self.encodec = EncodecModel.from_pretrained(config.encodec_model_name)
        self.transformer = MusicgenDecoder.from_pretrained(config.musicgen_decoder_model_name)

        self.musicgen_embedder = MusicGenEmbedder(self.encodec, self.transformer)

        # Add a projection head on top of the transformer encoder
        input_dim = self.transformer.config.hidden_size
        
        self.projection = MLPProjection(input_dim=input_dim, output_dim=config.projection_dim, hidden_dim=config.hidden_dim)        

        # Freeze the EnCodec and MusicGen model weights if required
        self.freeze_musicgen = config.freeze_musicgen
        if self.freeze_musicgen:
            for param in self.encodec.parameters():
                param.requires_grad = False
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Cache for storing computed embeddings
        self.embedding_cache = {}

    def forward(self, ids: Optional[List[str]], waveform: torch.Tensor, **kwargs):
        # Check if caching is applicable
        use_cache = self.freeze_musicgen and ids is not None

        if use_cache and all(id in self.embedding_cache for id in ids):
            batch_embeddings = torch.stack([self.embedding_cache[id] for id in ids])  # Shape: (B, 1024)
        else:
            with torch.no_grad() if self.freeze_musicgen else torch.enable_grad():
                batch_embeddings = self.musicgen_embedder.embed(waveform)  # Shape: (B, 1024)

            # Cache each sample separately only if IDs are provided
            if use_cache:
                for i, sample_id in enumerate(ids):
                    self.embedding_cache[sample_id] = batch_embeddings[i]

        # Apply projection layer
        return self.projection(batch_embeddings)  # Shape: (B, projection_dim)
    \
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Note that encodec and transformer weights will be loaded from the checkpoints saved by `save_pretrained` (so if we finetune their weights they will be loaded correctly), 
        # even if the constructor will first initiate them from its pretrained path, it will then be overriden since our save_pretrained save all weights 
        # inclduing the sub-models.
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if model.config.freeze_musicgen:
            for param in model.encodec.parameters():
                param.requires_grad = False
            for param in model.transformer.parameters():
                param.requires_grad = False
        return model