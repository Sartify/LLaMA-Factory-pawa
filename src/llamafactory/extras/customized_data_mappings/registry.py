from typing import Callable, Optional

from .data_mappings_pt import (
    build_swahili_english,
    build_xlam_function_calling_data_mapping,
)


PT_REGISTRY: dict[str, Optional[Callable]] = {
    "xlam-function-calling-60k-sharegpt": build_xlam_function_calling_data_mapping(),
    "swahili_english": build_swahili_english(),
}

SFT_REGISTRY: dict[str, Optional[Callable]] = {}
RM_REGISTRY: dict[str, Optional[Callable]] = {}

# stage: Literal["pt", "sft", "rm", "ppo", "kto"],
#

ALL_REGISTRYS: dict[str, dict[str, Optional[Callable]]] = {
    "pt": PT_REGISTRY,
    "sft": SFT_REGISTRY,
    "rm": RM_REGISTRY,
}
