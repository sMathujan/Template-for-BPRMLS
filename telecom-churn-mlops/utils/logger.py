"""One place to configure logging so every module logs consistently."""
import logging

from utils.config import get_logging_config

_CONFIGURED = False


def setup_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    cfg = get_logging_config()
    logging.basicConfig(
        level=getattr(logging, cfg.get("level", "INFO")),
        format=cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    )
    # Third-party libraries are noisy at INFO
    for noisy in ("mlflow", "optuna", "urllib3", "alembic", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def banner(logger: logging.Logger, title: str, width: int = 60) -> None:
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)
