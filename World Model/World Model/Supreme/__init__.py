"""Supreme -- the nested-learning memory core.

Hope (Behrouz et al., NeurIPS 2025, Section 8.3): a self-modifying Titans block
followed by a Continuum Memory System chain, dropped into the M model in place
of the MDN-RNN's LSTM.

Registered backbones:

``supreme``          Titans + CMS -- Hope as published
``supreme-titans``   Titans alone
``supreme-cms``      LSTM + CMS

Importing this package registers all three; nothing else in the repository
changes.  See ``README.md`` here for the equations and the design decisions,
including the two places where the paper is ambiguous and how each was resolved.
"""
from .cms import CMSBlock, ContinuumMemorySystem
from .hope import HopeBackbone, HopeState
from .titans import SelfModifyingTitans, TitansState

__all__ = [
    "HopeBackbone", "HopeState",
    "SelfModifyingTitans", "TitansState",
    "ContinuumMemorySystem", "CMSBlock",
]
