"""Baseline model: World Models (Ha & Schmidhuber, 2018).

    V  ConvVAE, 32-d latent            -> wmcore/vision/conv_vae.py   (shared)
    M  LSTM(256) + MDN head            -> lstm_backbone.py + wmcore MDN head
    C  linear controller, CMA-ES       -> wmcore/controller/           (shared)

Only the LSTM core lives here.  Everything else is shared with Supreme.
"""
from .attention_backbone import AttentionBackbone
from .lstm_backbone import GRUBackbone, LSTMBackbone

__all__ = ["LSTMBackbone", "GRUBackbone", "AttentionBackbone"]
