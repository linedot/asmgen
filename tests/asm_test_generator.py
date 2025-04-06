"""
Base class for ASM test generators
"""
from .generators.vec import vec_test_generator
from .generators.greg import greg_test_generator

class asm_test_generator(vec_test_generator, greg_test_generator):
    """
    Base class for ASM test generators, providing functionality by
    inheriting from specific test generators
    """
