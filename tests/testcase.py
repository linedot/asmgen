"""
Type hinting helper for asmgen test cases
"""

from asmgen.asmblocks.noarch import asmgen

# This exists just for type hinting
class testcase:
    """
    Type hinting helper for asmgen test cases
    """
    gen : asmgen
    cxx_name : str
    name : str
    allowed : list[str]
