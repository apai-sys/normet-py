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


class CalibrationWarning(UserWarning):
    """A calibrated band still fails to cover at its nominal rate.

    Emitted by :func:`normet.calibrate_quantiles` when the hold-out's
    cross-validated coverage of a calibrated band deviates from the nominal
    level by more than ten points: the correction was fitted, but the
    record's behaviour is beyond it (a shift or spread the hold-out never
    showed). The band is returned as corrected and labelled --- the band
    that looks calibrated while its coverage is not is the one this warning
    exists to keep out of a paper.

    To suppress::

        import warnings
        warnings.filterwarnings("ignore", category=normet.CalibrationWarning)
    """
