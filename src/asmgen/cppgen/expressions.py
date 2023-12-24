from enum import Enum,unique

def identity(c_var):
    return f"({c_var})"

def add(c_var1, c_var2):
    return f"({c_var1}+{c_var2})"

def mul(c_var1, c_var2):
    return f"({c_var1}*{c_var2})"

def div(c_var1, c_var2):
    return f"({c_var1}/{c_var2})"

@unique
class shift_direction(Enum):
    LEFT=0
    RIGHT=1

class shiftlit(object):
    def __init__(self, direction, bit_count=1):
        self.shiftop = "<<" if (direction == shift_direction.LEFT) else ">>"
        self.bit_count = bit_count
    def __call__(self,c_var):
        return f"({c_var} {self.shiftop} {self.bit_count})"

class shiftvar(object):
    def __init__(self, direction):
        self.shiftop = "<<" if (direction == shift_direction.LEFT) else ">>"
    def __call__(self, c_var1, c_var2):
        return f"({c_var1} {self.shiftop} {c_var2})"
