# lang_guardx/__init__.py
from .agent import PolicyVerdict, SQLPolicy, SQLPolicyEngine, Verdict
from .detection.core import DetectionResult, Detector

try:
    from .agent import AgentTrace, ProtectedSQLAgent, StepTrace
except ImportError:
    pass

__version__ = "0.1.2"

__all__ = [
    "Detector",
    "DetectionResult",
    "SQLPolicy",
    "PolicyVerdict",
    "Verdict",
    "SQLPolicyEngine",
    "ProtectedSQLAgent",
    "AgentTrace",
    "StepTrace",
]
