from ...registers import vreg

class sve_vreg(vreg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"z{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class sve_preg(vreg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"p{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str
