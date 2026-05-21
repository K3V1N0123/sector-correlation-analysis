"""Custom project exceptions."""


class DataIngestionError(Exception):
    """Raised when fewer than 10 sectors can be fetched."""


class StationarityError(Exception):
    """Raised when a series fails ADF after first differencing."""


class InsufficientDataError(Exception):
    """Raised when sub-period has fewer than minimum required rows."""


class OutputWriteError(Exception):
    """Raised when a required output cannot be saved."""
