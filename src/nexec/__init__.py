"""nexec: no-execute bits for LLM tokens.

Mark the tokens that came from outside (tool output, retrieved documents, emails) and nexec removes their
instruction-like component inside the model (NXDirection), so the model reads them as data instead of obeying them.
"""

from nexec._version import __version__
from nexec.directions import Directions, DirectionsConfig, DirectionsMismatchError
from nexec.hook import protect
from nexec.nx import NX
from nexec.provenance import Encoded, Rendered, TemplateError, encode, render, untrusted

__all__ = ["NX", "Directions", "DirectionsConfig", "DirectionsMismatchError", "Encoded", "Rendered",
           "TemplateError", "__version__", "encode", "protect", "render", "untrusted"]
