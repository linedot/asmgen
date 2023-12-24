from abc import ABC, abstractmethod

from asmgen.asmblocks.noarch import asmgen, reg_tracker
from asmgen.cppgen.declarations import vargen



# This exists just for type hinting
class testcase(ABC):
    gen : asmgen
    cxx_name : str
    name : str

    @classmethod
    @abstractmethod
    def add_test(cls, name : str,
                 rt : reg_tracker,
                 vg : vargen,
                 asmblock : str,
                 check_function_definition : str,
                 check : str,
                 extra_prepare : str =""):
        raise NotImplementedError("To be implemented by test case")

    @staticmethod
    @abstractmethod
    def random_immediate():
        raise NotImplementedError("To be implemented by test case")
