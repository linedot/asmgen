from asmgen.registers import reg_tracker
from asmgen.cppgen.checkers import bi_u64_eq
from asmgen.cppgen.declarations import vargen
from asmgen.cppgen.expressions import identity
import logging

from ..testcase import testcase

class greg_test_generator:
    ###################################
    # Test code generators start here #
    ###################################

    @classmethod
    def generate_mov_imm_to_greg(cls, tcase : testcase,
                                 rt : reg_tracker, vg : vargen):
        log = logging.getLogger("ASMCORTEST")
        test_name = "mov_imm_to_greg"

        varname = vg.new_var("std::uint64_t")
        reg_idx = rt.reserve_any_reg(type_tag="greg")
        reg = tcase.gen.greg(reg_idx)
        imm = tcase.random_immediate()

        asmblock  = tcase.gen.zero_greg(reg)
        asmblock += tcase.gen.mov_greg_imm(reg, imm)
        asmblock += tcase.gen.mov_greg_to_param(reg, varname)

        rt.unuse_reg(type_tag="greg", idx=reg_idx)
        check_function = bi_u64_eq(f"check_{test_name}",identity)
        tcase.add_test(test_name, rt, vg,
                      asmblock, 
                      check_function,
                      f"check_{test_name}({varname}, {imm})")

    @classmethod
    def generate_add_imm_to_greg(cls, tcase : testcase, rt : reg_tracker, vg : vargen):
        log = logging.getLogger("ASMCORTEST")
        test_name = "add_imm_to_greg"

        varname = vg.new_var("std::uint64_t")
        reg_idx = rt.reserve_any_reg(type_tag="greg")
        reg = tcase.gen.greg(reg_idx)
        imm = tcase.random_immediate()
        asmblock  = tcase.gen.zero_greg(reg)
        asmblock += tcase.gen.add_greg_imm(reg, imm)
        asmblock += tcase.gen.mov_greg_to_param(reg, varname)
        rt.unuse_reg(type_tag="greg", idx=reg_idx)
        check_function = bi_u64_eq(f"check_{test_name}",identity)
        tcase.add_test(test_name, rt, vg,
                      asmblock, 
                      check_function,
                      f"check_{test_name}({varname}, {imm})")

    @classmethod
    def generate_add_imm_to_greg_2_times(cls, tcase : testcase, rt : reg_tracker, vg : vargen):
        log = logging.getLogger("ASMCORTEST")
        test_name = "add_imm_to_greg_2_times"

        varname = vg.new_var("std::uint64_t")
        reg_idx = rt.reserve_any_reg(type_tag="greg")
        reg = tcase.gen.greg(reg_idx)
        imm = tcase.random_immediate()
        asmblock  = tcase.gen.zero_greg(reg)
        asmblock += tcase.gen.add_greg_imm(reg, imm)
        asmblock += tcase.gen.add_greg_imm(reg, imm)
        asmblock += tcase.gen.mov_greg_to_param(reg, varname)
        rt.unuse_reg(type_tag="greg", idx=reg_idx)
        check_function = bi_u64_eq(f"check_{test_name}",identity)
        tcase.add_test(test_name, rt, vg,
                      asmblock, 
                      check_function,
                      f"check_{test_name}({varname}, {imm+imm})")

    @classmethod
    def generate_add_greg_to_greg(cls, tcase : testcase, rt : reg_tracker, vg : vargen):
        log = logging.getLogger("ASMCORTEST")
        test_name = "add_greg_to_greg"

        varname = vg.new_var("std::uint64_t")
        dst_idx = rt.reserve_any_reg(type_tag="greg")
        reg1_idx = rt.reserve_any_reg(type_tag="greg")
        reg2_idx = rt.reserve_any_reg(type_tag="greg")

        dst = tcase.gen.greg(dst_idx)
        reg1 = tcase.gen.greg(reg1_idx)
        reg2 = tcase.gen.greg(reg2_idx)

        imm = tcase.random_immediate()

        asmblock  = tcase.gen.zero_greg(dst)
        asmblock += tcase.gen.zero_greg(reg1)
        asmblock += tcase.gen.zero_greg(reg2)
        asmblock += tcase.gen.add_greg_imm(reg1, imm)
        asmblock += tcase.gen.add_greg_imm(reg2, imm)
        asmblock += tcase.gen.add_greg_greg(dst, reg1, reg2)
        asmblock += tcase.gen.mov_greg_to_param(dst, varname)

        rt.unuse_reg(type_tag="greg", idx=dst_idx)
        rt.unuse_reg(type_tag="greg", idx=reg1_idx)
        rt.unuse_reg(type_tag="greg", idx=reg2_idx)

        check_function = bi_u64_eq(f"check_{test_name}",identity)
        tcase.add_test(test_name, rt, vg,
                      asmblock, 
                      check_function,
                      f"check_{test_name}({varname}, {imm+imm})")
