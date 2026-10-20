"""Guard model inputs that are outside a checkpoint's documented benchmark scope."""
from collections.abc import Mapping
from typing import Any


def supports_high_resolution_tiles(training_config: Any) -> bool:
    """PathMNIST's 28×28 benchmark checkpoint is not a high-resolution tile model."""
    return not (
        isinstance(training_config, Mapping)
        and training_config.get("dataset") == "pathmnist"
    )
