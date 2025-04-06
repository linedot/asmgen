"""
Generator for tests of GP register-related ASM blocks
"""
from asmgen.registers import reg_tracker
from asmgen.cppgen.checkers import bi_u64_eq
from asmgen.cppgen.declarations import vargen
from asmgen.cppgen.expressions import identity

from ..testcase import testcase

class greg_test_generator:
    """
    Generator of tests of operations on GP register
    """
    ###################################
    # Test code generators start here #
    ###################################

    @classmethod
    def generate_mov_imm_to_greg(cls, tcase : testcase,
                                 rt : reg_tracker, vg : vargen):
        """
        Tests correctness of an ASM block that moves an immediate into a GP register
        """
        test_name = "mov_imm_to_greg"

        varname = vg.new_var(cpp_type="std::uint64_t")
        reg_idx = rt.reserve_any_reg(type_tag="greg")
        reg = tcase.gen.greg(reg_idx=reg_idx)
        imm = tcase.random_immediate()

        asmblock  = tcase.gen.zero_greg(greg=reg)
        asmblock += tcase.gen.mov_greg_imm(reg=reg, imm=imm)
        asmblock += tcase.gen.mov_greg_to_param(src=reg, param=varname)

        rt.unuse_reg(type_tag="greg", idx=reg_idx)
        check_function = bi_u64_eq(f"check_{test_name}",identity)
        tcase.add_test(name=test_name, rt=rt, vg=vg,
                       asmblock=asmblock,
                       check_function_definition=check_function,
                       check=f"check_{test_name}({varname}, {imm})")

    @classmethod
    def generate_add_imm_to_greg(cls, tcase : testcase, rt : reg_tracker, vg : vargen):
        """
        Tests correctness of an ASM block that adds an immediate to a GP register
        """
        test_name = "add_imm_to_greg"

        varname = vg.new_var(cpp_type="std::uint64_t")
        reg_idx = rt.reserve_any_reg(type_tag="greg")
        reg = tcase.gen.greg(reg_idx)
        imm = tcase.random_immediate()
        asmblock  = tcase.gen.zero_greg(greg=reg)
        asmblock += tcase.gen.add_greg_imm(reg=reg, imm=imm)
        asmblock += tcase.gen.mov_greg_to_param(src=reg, param=varname)
        rt.unuse_reg(type_tag="greg", idx=reg_idx)
        check_function = bi_u64_eq(f"check_{test_name}",identity)
        tcase.add_test(name=test_name, rt=rt, vg=vg,
                       asmblock=asmblock,
                       check_function_definition=check_function,
                       check=f"check_{test_name}({varname}, {imm})")

    @classmethod
    def generate_add_imm_to_greg_2_times(cls, tcase : testcase, rt : reg_tracker, vg : vargen):
        """
        Tests correctness of an ASM block that adds an immediate to a GP register 2 times
        """
        test_name = "add_imm_to_greg_2_times"

        varname = vg.new_var(cpp_type="std::uint64_t")
        reg_idx = rt.reserve_any_reg(type_tag="greg")
        reg = tcase.gen.greg(reg_idx)
        imm = tcase.random_immediate()
        asmblock  = tcase.gen.zero_greg(greg=reg)
        asmblock += tcase.gen.add_greg_imm(reg=reg, imm=imm)
        asmblock += tcase.gen.add_greg_imm(reg=reg, imm=imm)
        asmblock += tcase.gen.mov_greg_to_param(src=reg, param=varname)
        rt.unuse_reg(type_tag="greg", idx=reg_idx)
        check_function = bi_u64_eq(f"check_{test_name}",identity)
        tcase.add_test(name=test_name, rt=rt, vg=vg,
                      asmblock=asmblock,
                      check_function_definition=check_function,
                      check=f"check_{test_name}({varname}, {imm+imm})")

    @classmethod
    def generate_add_greg_to_greg(cls, tcase : testcase, rt : reg_tracker, vg : vargen):
        """
        Tests correctness of an ASM block that adds the value of a GP register
        to another GP register
        """
        test_name = "add_greg_to_greg"

        varname = vg.new_var(cpp_type="std::uint64_t")
        dst_idx = rt.reserve_any_reg(type_tag="greg")
        reg1_idx = rt.reserve_any_reg(type_tag="greg")
        reg2_idx = rt.reserve_any_reg(type_tag="greg")

        dst = tcase.gen.greg(dst_idx)
        reg1 = tcase.gen.greg(reg1_idx)
        reg2 = tcase.gen.greg(reg2_idx)

        imm = tcase.random_immediate()

        asmblock  = tcase.gen.zero_greg(greg=dst)
        asmblock += tcase.gen.zero_greg(greg=reg1)
        asmblock += tcase.gen.zero_greg(greg=reg2)
        asmblock += tcase.gen.add_greg_imm(reg=reg1, imm=imm)
        asmblock += tcase.gen.add_greg_imm(reg=reg2, imm=imm)
        asmblock += tcase.gen.add_greg_greg(dst=dst, reg1=reg1, reg2=reg2)
        asmblock += tcase.gen.mov_greg_to_param(src=dst, param=varname)

        rt.unuse_reg(type_tag="greg", idx=dst_idx)
        rt.unuse_reg(type_tag="greg", idx=reg1_idx)
        rt.unuse_reg(type_tag="greg", idx=reg2_idx)

        check_function = bi_u64_eq(f"check_{test_name}",identity)
        tcase.add_test(name=test_name, rt=rt, vg=vg,
                      asmblock=asmblock,
                      check_function_definition=check_function,
                      check=f"check_{test_name}({varname}, {imm+imm})")
