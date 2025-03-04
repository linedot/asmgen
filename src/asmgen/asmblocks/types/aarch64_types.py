from ...registers import greg,freg

class aarch64_greg(greg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"x{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class aarch64_freg(freg):
    def __init__(self, reg_idx : int):
        # TODO: Other data types
        self.reg_str = f"d{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str
