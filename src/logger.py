# Standard libraries
import sys

# Third-party libraries
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format=(
        "<green><level>GLIOMOL:"
        " </level></green><blue><level>{message}</level></blue>"
    ),
)