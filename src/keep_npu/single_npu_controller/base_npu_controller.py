from typing import Optional, Union

from keep_npu.utilities.humanized_input import parse_vram_to_elements
from keep_npu.utilities.session_config import (
    normalize_utilization_percent,
    validate_interval,
)


class BaseNPUController:
    def __init__(self, vram_to_keep: Union[int, str], interval: float):
        self.vram_to_keep = parse_vram_to_elements(vram_to_keep)
        self.interval = validate_interval(interval)

    @staticmethod
    def _should_run_batch(
        npu_utilization: Optional[int], busy_threshold: int
    ) -> bool:
        if busy_threshold < 0:
            return True
        utilization = normalize_utilization_percent(npu_utilization)
        return utilization is not None and utilization <= busy_threshold

