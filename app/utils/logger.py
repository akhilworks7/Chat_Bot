import logging
import sys
import colorlog


def get_logger(name: str = "rag_chatbot") -> logging.Logger:
    """
    Returns a configured logger instance with colored console output.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


logger = get_logger("app")
