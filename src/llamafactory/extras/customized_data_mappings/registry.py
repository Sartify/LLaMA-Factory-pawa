from typing import Dict, Type, Callable, Optional


PT_REGISTRY: dict[str, Optional[Callable]] = {}
SFT_REGISTRY: dict[str, Optional[Callable]] = {}
RM_REGISTRY: dict[str, Optional[Callable]] = {}

ALL_REGISTRYS: dict[str, dict[str, Optional[Callable]]] = {
    "pt": PT_REGISTRY,
    "sft": SFT_REGISTRY,
    "rm": RM_REGISTRY,
}
# stage: Literal["pt", "sft", "rm", "ppo", "kto"],
