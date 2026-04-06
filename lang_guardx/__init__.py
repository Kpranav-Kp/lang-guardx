# lang_guardx/__init__.py
from .detection.core import Detector, DetectionResult
from .agent import SQLPolicy, PolicyVerdict, Verdict, SQLPolicyEngine

try:
    from .agent import ProtectedSQLAgent, AgentTrace, StepTrace
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