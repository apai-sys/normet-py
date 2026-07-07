"""Custom exception hierarchy for normet."""


class NormetError(Exception):
    """Base exception for all normet errors."""


class DataError(NormetError):
    """Invalid, missing, or malformed input data."""


class ModelError(NormetError):
    """Model training, prediction, or persistence failure."""


class ConfigError(NormetError):
    """Invalid configuration or missing required parameters."""


class ExperimentalWarning(UserWarning):
    """Raised when an experimental feature is used.

    To suppress::

        import warnings
        warnings.filterwarnings("ignore", category=normet.ExperimentalWarning)
    """
