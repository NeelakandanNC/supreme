"""
wmcore
======

Shared, model-agnostic core of the World-Model comparison study.

Everything in this package is *identical* for both models under comparison:

    Baseline ("World Model")  = V (ConvVAE)  +  M (LSTM backbone)   + C (linear)
    Ours     ("Supreme")      = V (ConvVAE)  +  M (nested-learning) + C (linear)

The **only** thing that differs between the two is the recurrent backbone that
lives inside the memory module (``wmcore.memory.base.MemoryBackbone``).
Concrete backbones live under ``models/baseline`` and ``models/supreme``.

Keeping V, C, the environments, the data pipeline, the output head, the loss,
the optimiser and the evaluation harness in one shared package is what makes
the comparison apples-to-apples: there is exactly one axis of variation, and it
is the axis the paper is about.
"""

__version__ = "0.1.0"
