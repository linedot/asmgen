from ...registers import vreg

class rvv_vreg(vreg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"v{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str
