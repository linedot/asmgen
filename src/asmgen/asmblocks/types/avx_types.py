from ...registers import vreg,greg,freg

from typing import Union

class x86_greg(greg):
    
    greg_names = [f'r{i}' for i in [str(j) for j in range(8,16)]+['ax','bx','cx','dx','si','di','bp','sp']]

    def __init__(self, reg_idx : int):
        self.reg_str = avxbase.greg_names[reg_idx]

    def __str__(self) -> str:
        return self.reg_str

class avx_freg(freg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"xmm{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class xmm_vreg(vreg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"xmm{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class ymm_vreg(vreg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"ymm{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class zmm_vreg(vreg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"zmm{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

def prefix_if_raw_reg(reg : Union[greg,freg,vreg]) -> str:
    # If there is a [ it's probably a parameter
    if '[' in str(reg):
        return str(reg)
    return f"%%{reg}"
