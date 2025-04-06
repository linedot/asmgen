"""
NEON/ASIMD register types
"""
from ...registers import vreg_base

#pylint: disable=too-few-public-methods
class neon_vreg(vreg_base):
    """
    NEON/ASIMD vector register
    """
    def __init__(self, reg_idx : int):
        self.idx = reg_idx
    def __str__(self) -> str:
        return f"v{self.idx}"
