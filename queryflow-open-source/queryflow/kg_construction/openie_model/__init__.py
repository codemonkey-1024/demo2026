from .base_model import BaseOPENIEModel
try:
    from .llm_openie_model import LLMOPENIEModel
except Exception:
    LLMOPENIEModel = None

__all__ = ["BaseOPENIEModel"]
if LLMOPENIEModel is not None:
    __all__.append("LLMOPENIEModel")
