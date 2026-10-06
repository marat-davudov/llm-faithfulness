import logging


def get_logger(name: str) -> logging.Logger:
    """Configure the root logging format (once) and return a named logger."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(name)
