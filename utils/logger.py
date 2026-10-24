import logging
import sys
import os
from typing import Optional

def setup_logger(
    name: str = "ecg_framework",
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """Configures a standardized logging pipeline with support for Rich console output.

    If the `rich` package is installed, it integrates RichHandler for modern
    interactive outputs. Otherwise, it falls back to a clean stream-logger.

    Args:
        name: Name of the logger instances.
        log_level: Severity thresholds, e.g. "DEBUG", "INFO", "WARNING", "ERROR".
        log_file: Path to write raw log outputs to.

    Returns:
        logging.Logger: The configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    # Log Formatter setup
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Rich logger check
    use_rich = False
    try:
        from rich.logging import RichHandler
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_path=False
        )
        use_rich = True
    except ImportError:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.addHandler(console_handler)

    # Log to file handler if filepath provided
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File captures verbose DEBUG entries
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f"Initialized logger '{name}' (Rich terminal logs: {use_rich})")
    return logger
