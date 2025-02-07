import numpy as np
import torch
from torch.nn import Parameter
from transformers import PreTrainedModel, PretrainedConfig

from src.modules.clap_audio_encoder import ClapMusicGenAudioEncoder, ClapMusicGenAudioEncoderConfig
from src.modules.clap_text_encoder import ClapTextEncoder, ClapTextEncoderConfig

class CLAPConfig(PretrainedConfig):
    def __init__(self, 
                 audio_encoder_config: ClapMusicGenAudioEncoderConfig = None, 
                 text_encoder_config: ClapTextEncoderConfig = None,
                 temp_a: float = 0.05,
                 temp_t: float = 0.05,
                 **kwargs):
        super().__init__(**kwargs)
        self.audio_encoder_config = audio_encoder_config or {}
        self.text_encoder_config = text_encoder_config or {}
        self.temp_a = temp_a
        self.temp_t = temp_t

class CLAPModel(PreTrainedModel):
    config_class = CLAPConfig

    def __init__(self, config):
        super().__init__(config)
        
        self.audio_encoder = ClapMusicGenAudioEncoder(
            ClapMusicGenAudioEncoderConfig(**config.audio_encoder_config)
        )
        self.text_encoder = ClapTextEncoder(
            ClapTextEncoderConfig(**config.text_encoder_config)
        )

        # Logit scaling factors as learnable parameters
        self.logit_scale_a = Parameter(torch.ones([]) * np.log(1 / config.temp_a))
        self.logit_scale_t = Parameter(torch.ones([]) * np.log(1 / config.temp_t))

    def forward(self, ids, audio_waveforms, captions):
        proj_audio_embeds = self.audio_encoder(ids, audio_waveforms)
        proj_text_embeds = self.text_encoder(ids, **captions)

        # detach pretrained text embeddings from the computation graph to prevent backprop to text encoder
        pretrained_text_embeds = self.text_encoder.get_pretrained_text_emb(ids, **captions).detach()

        # Dynamically clamp logit scales during forward pass
        logit_scale_a = torch.clamp(self.logit_scale_a, max=2.0)
        logit_scale_t = torch.clamp(self.logit_scale_t, max=2.0)

        return proj_audio_embeds, proj_text_embeds, pretrained_text_embeds, logit_scale_a, logit_scale_t


def init_clap_model(freeze_musicgen=True, freeze_text_encoder=True, projection_dim=1024, hidden_dim=512, temp_a=0.01, temp_t=0.01):
    # Initialize configuration
    config = CLAPConfig(
        audio_encoder_config={
            "encodec_model_name": "model/encodec", 
            "musicgen_decoder_model_name": "model/musicgen_decoder", 
            "projection_dim": projection_dim, 
            "hidden_dim": hidden_dim,
            "freeze_musicgen": freeze_musicgen
        },
        text_encoder_config={            
            "projection_dim": projection_dim, 
            "hidden_dim": hidden_dim,
            "freeze_text_encoder": freeze_text_encoder
        },
        temp_a=temp_a,
        temp_t=temp_t
    )

    model = CLAPModel(config)
    print(f'Initialized model, freeze_musicgen: {freeze_musicgen}, freeze_text_encoder: {freeze_text_encoder}, projection_dim: {projection_dim}, hidden_dim: {hidden_dim}, temp_a: {temp_a}, temp_t: {temp_t}')
    return model
