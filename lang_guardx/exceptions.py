from __future__ import annotations


class LangGuardXError(Exception):
    """Base exception for all LangGuardX errors."""


class ConfigurationError(LangGuardXError):
    """Invalid or missing configuration."""


class DetectionError(LangGuardXError):
    """Error during input detection."""


class PolicyViolation(LangGuardXError):
    """SQL policy violation detected.

    Reserved for custom rules or future middleware that prefer
    exceptions over ``PolicyVerdict`` returns.
    """


class BlockedRequest(LangGuardXError):
    """Request was blocked by LangGuardX."""


class PluginRegistrationError(LangGuardXError):
    """Error registering a plugin/layer."""


class AdaptationError(LangGuardXError):
    """Error during runtime adaptation.

    Reserved for custom adaptive backends that raise on failure.
    """
