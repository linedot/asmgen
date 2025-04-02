from asmgen.asmblocks.noarch import asm_data_type as dt, reg_tracker
from asmgen.cppgen.checkers import always_true, vec_fp64_close, vec_fp32_close
from asmgen.cppgen.expressions import identity
from asmgen.cppgen.declarations import vargen, vio_type
import logging

class vec_test_generator:
    @classmethod
    def generate_mov_fp32_to_vreg(cls, testcase, rt : reg_tracker, vg : vargen):
        log = logging.getLogger("ASMCORTEST")
        test_name = "mov_fp32_to_vreg"
        imm = testcase.random_fimmediate()
        varname = vg.new_vector("float",
                                "get_simd_size()/sizeof(float)",
                                imm,
                                vt=vio_type.INPUT)


        asmblock = ""

        asmblock += testcase.gen.isaquirks(rt, dt.SINGLE)

        areg_idx = rt.reserve_any_reg(type_tag="greg")
        areg = testcase.gen.greg(areg_idx)
        asmblock += testcase.gen.zero_greg(areg)
        asmblock += testcase.gen.mov_param_to_greg(varname, areg)

        vreg_idx = rt.reserve_any_reg(type_tag="vreg")
        vreg = testcase.gen.vreg(vreg_idx)
        asmblock += testcase.gen.zero_vreg(vreg, dt.SINGLE)
        asmblock += testcase.gen.load_vector(areg, 0, vreg, dt.SINGLE)

        rt.unuse_reg(type_tag="greg", idx=areg_idx)
        rt.unuse_reg(type_tag="vreg", idx=vreg_idx)

        check_function = always_true(f"check_{test_name}")
        testcase.add_test(test_name, rt, vg,
                      asmblock, 
                      check_function,
                      f"check_{test_name}()")

    @classmethod
    def generate_mov_fp64_to_vreg(cls, testcase, rt : reg_tracker, vg : vargen):
        log = logging.getLogger("ASMCORTEST")
        test_name = "mov_fp64_to_vreg"
        imm = testcase.random_fimmediate()
        varname = vg.new_vector("double",
                                "get_simd_size()/sizeof(double)",
                                imm,
                                vt=vio_type.INPUT)


        asmblock = ""
        asmblock += testcase.gen.isaquirks(rt, dt.DOUBLE)

        areg_idx = rt.reserve_any_reg(type_tag="greg")
        areg = testcase.gen.greg(areg_idx)
        asmblock += testcase.gen.zero_greg(areg)
        asmblock += testcase.gen.mov_param_to_greg(varname, areg)

        vreg_idx = rt.reserve_any_reg(type_tag="vreg")
        vreg = testcase.gen.vreg(vreg_idx)
        asmblock += testcase.gen.zero_vreg(vreg, dt.DOUBLE)
        asmblock += testcase.gen.load_vector(areg, 0, vreg, dt.DOUBLE)

        rt.unuse_reg(type_tag="greg", idx=areg_idx)
        rt.unuse_reg(type_tag="vreg", idx=vreg_idx)

        check_function = always_true(f"check_{test_name}")
        testcase.add_test(test_name, rt, vg,
                      asmblock, 
                      check_function,
                      f"check_{test_name}()")

    @classmethod
    def generate_vec_fma_fp32(cls, testcase, rt : reg_tracker, vg : vargen):
        log = logging.getLogger("ASMCORTEST")
        test_name = "vec_fma_fp32"

        simd_nelements = "get_simd_size()/sizeof(float)"

        imm1 = testcase.random_fimmediate()
        avec = vg.new_vector("float",
                             simd_nelements,
                             imm1,
                             vt=vio_type.INPUT)
        imm2 = testcase.random_fimmediate()
        bvec = vg.new_vector("float",
                             simd_nelements,
                             imm2,
                             vt=vio_type.INPUT)
        imm3 = testcase.random_fimmediate()
        cvec = vg.new_vector("float",
                             simd_nelements,
                             imm3,
                             vt=vio_type.INPUT)

        avreg_idx = rt.reserve_any_reg(type_tag="vreg")
        avreg = testcase.gen.vreg(avreg_idx)
        bvreg_idx = rt.reserve_any_reg(type_tag="vreg")
        bvreg = testcase.gen.vreg(bvreg_idx)
        cvreg_idx = rt.reserve_any_reg(type_tag="vreg")
        cvreg = testcase.gen.vreg(cvreg_idx)

        asmblock  = testcase.gen.isaquirks(rt, dt.SINGLE)

        asmblock += testcase.gen.zero_vreg(avreg, dt.SINGLE)
        asmblock += testcase.gen.zero_vreg(bvreg, dt.SINGLE)
        asmblock += testcase.gen.zero_vreg(cvreg, dt.SINGLE)


        aareg_idx = rt.reserve_any_reg(type_tag="greg")
        aareg = testcase.gen.greg(aareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(avec, aareg)
        asmblock += testcase.gen.load_vector(aareg, 0, avreg, dt.SINGLE)
        rt.unuse_reg(type_tag="greg", idx=aareg_idx)

        bareg_idx = rt.reserve_any_reg(type_tag="greg")
        bareg = testcase.gen.greg(bareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(bvec, bareg)
        asmblock += testcase.gen.load_vector(bareg, 0, bvreg, dt.SINGLE)
        rt.unuse_reg(type_tag="greg", idx=bareg_idx)

        careg_idx = rt.reserve_any_reg(type_tag="greg")
        careg = testcase.gen.greg(careg_idx)
        asmblock += testcase.gen.mov_param_to_greg(cvec, careg)
        asmblock += testcase.gen.load_vector(careg, 0, cvreg, dt.SINGLE)
        asmblock += testcase.gen.fma(avreg, bvreg, cvreg, dt.SINGLE, dt.SINGLE, dt.SINGLE)
        asmblock += testcase.gen.store_vector(careg, 0, cvreg, dt.SINGLE)
        rt.unuse_reg(type_tag="greg", idx=careg_idx)

        rt.unuse_reg(type_tag="vreg", idx=avreg_idx)
        rt.unuse_reg(type_tag="vreg", idx=bvreg_idx)
        rt.unuse_reg(type_tag="vreg", idx=cvreg_idx)

        extra_prepare  = f"std::vector<float> result({simd_nelements});\n"
        # adding up in python will actually result in failures
        extra_prepare += f"std::fill(result.begin(), result.end(), {imm3}+{imm1}*{imm2});\n"

        check_function = vec_fp32_close(f"check_{test_name}", identity)
        testcase.add_test(test_name, rt, vg,
                      asmblock, 
                      check_function,
                      f"check_{test_name}({cvec}, result)",
                      extra_prepare=extra_prepare)

    @classmethod
    def generate_vec_fma_fp64(cls, testcase, rt : reg_tracker, vg : vargen):
        log = logging.getLogger("ASMCORTEST")
        test_name = "vec_fma_fp64"

        simd_nelements = "get_simd_size()/sizeof(double)"

        imm1 = testcase.random_fimmediate()
        avec = vg.new_vector("double",
                             simd_nelements,
                             imm1,
                             vt=vio_type.INPUT)
        imm2 = testcase.random_fimmediate()
        bvec = vg.new_vector("double",
                             simd_nelements,
                             imm2,
                             vt=vio_type.INPUT)
        imm3 = testcase.random_fimmediate()
        cvec = vg.new_vector("double",
                             simd_nelements,
                             imm3,
                             vt=vio_type.INPUT)

        avreg_idx = rt.reserve_any_reg(type_tag="vreg")
        avreg = testcase.gen.vreg(avreg_idx)
        bvreg_idx = rt.reserve_any_reg(type_tag="vreg")
        bvreg = testcase.gen.vreg(bvreg_idx)
        cvreg_idx = rt.reserve_any_reg(type_tag="vreg")
        cvreg = testcase.gen.vreg(cvreg_idx)

        asmblock  = testcase.gen.isaquirks(rt, dt.DOUBLE)

        asmblock += testcase.gen.zero_vreg(avreg, dt.DOUBLE)
        asmblock += testcase.gen.zero_vreg(bvreg, dt.DOUBLE)
        asmblock += testcase.gen.zero_vreg(cvreg, dt.DOUBLE)


        aareg_idx = rt.reserve_any_reg(type_tag="greg")
        aareg = testcase.gen.greg(aareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(avec, aareg)
        asmblock += testcase.gen.load_vector(aareg, 0, avreg, dt.DOUBLE)
        rt.unuse_reg(type_tag="greg", idx=aareg_idx)

        bareg_idx = rt.reserve_any_reg(type_tag="greg")
        bareg = testcase.gen.greg(bareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(bvec, bareg)
        asmblock += testcase.gen.load_vector(bareg, 0, bvreg, dt.DOUBLE)
        rt.unuse_reg(type_tag="greg", idx=bareg_idx)

        careg_idx = rt.reserve_any_reg(type_tag="greg")
        careg = testcase.gen.greg(careg_idx)
        asmblock += testcase.gen.mov_param_to_greg(cvec, careg)
        asmblock += testcase.gen.load_vector(careg, 0, cvreg, dt.DOUBLE)
        asmblock += testcase.gen.fma(avreg, bvreg, cvreg, dt.DOUBLE, dt.DOUBLE, dt.DOUBLE)
        asmblock += testcase.gen.store_vector(careg, 0, cvreg, dt.DOUBLE)
        rt.unuse_reg(type_tag="greg", idx=careg_idx)

        rt.unuse_reg(type_tag="vreg", idx=avreg_idx)
        rt.unuse_reg(type_tag="vreg", idx=bvreg_idx)
        rt.unuse_reg(type_tag="vreg", idx=cvreg_idx)

        extra_prepare  = f"std::vector<double> result({simd_nelements});\n"
        # adding up in python will actually result in failures
        extra_prepare += f"std::fill(result.begin(), result.end(), {imm3}+{imm1}*{imm2});\n"

        check_function = vec_fp64_close(f"check_{test_name}", identity)
        testcase.add_test(test_name, rt, vg,
                      asmblock, 
                      check_function,
                      f"check_{test_name}({cvec}, result)",
                      extra_prepare=extra_prepare)

    @classmethod
    def generate_vec_fmul_fp32(cls, testcase, rt : reg_tracker, vg : vargen):
        log = logging.getLogger("ASMCORTEST")
        test_name = "vec_fmul_fp32"

        simd_nelements = "get_simd_size()/sizeof(float)"

        imm1 = testcase.random_fimmediate()
        avec = vg.new_vector("float",
                             simd_nelements,
                             imm1,
                             vt=vio_type.INPUT)
        imm2 = testcase.random_fimmediate()
        bvec = vg.new_vector("float",
                             simd_nelements,
                             imm2,
                             vt=vio_type.INPUT)

        avreg_idx = rt.reserve_any_reg(type_tag="vreg")
        avreg = testcase.gen.vreg(avreg_idx)
        bvreg_idx = rt.reserve_any_reg(type_tag="vreg")
        bvreg = testcase.gen.vreg(bvreg_idx)

        asmblock  = testcase.gen.isaquirks(rt, dt.SINGLE)

        asmblock += testcase.gen.zero_vreg(avreg, dt.SINGLE)
        asmblock += testcase.gen.zero_vreg(bvreg, dt.SINGLE)

        aareg_idx = rt.reserve_any_reg(type_tag="greg")
        aareg = testcase.gen.greg(aareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(avec, aareg)
        asmblock += testcase.gen.load_vector(aareg, 0, avreg, dt.SINGLE)

        bareg_idx = rt.reserve_any_reg(type_tag="greg")
        bareg = testcase.gen.greg(bareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(bvec, bareg)
        asmblock += testcase.gen.load_vector(bareg, 0, bvreg, dt.SINGLE)
        rt.unuse_reg(type_tag="greg", idx=bareg_idx)

        asmblock += testcase.gen.fmul(avreg, bvreg, avreg, dt.SINGLE, dt.SINGLE, dt.SINGLE)
        asmblock += testcase.gen.store_vector(aareg, 0, avreg, dt.SINGLE)
        rt.unuse_reg(type_tag="greg", idx=aareg_idx)

        rt.unuse_reg(type_tag="vreg", idx=avreg_idx)
        rt.unuse_reg(type_tag="vreg", idx=bvreg_idx)

        extra_prepare  = f"std::vector<float> result({simd_nelements});\n"
        # adding up in python will actually result in failures
        extra_prepare += f"std::fill(result.begin(), result.end(), {imm1}*{imm2});\n"

        check_function = vec_fp32_close(f"check_{test_name}", identity)
        testcase.add_test(test_name, rt, vg,
                      asmblock, 
                      check_function,
                      f"check_{test_name}({avec}, result)",
                      extra_prepare=extra_prepare)

    @classmethod
    def generate_vec_fmul_fp64(cls, testcase, rt : reg_tracker, vg : vargen):
        log = logging.getLogger("ASMCORTEST")
        test_name = "vec_fmul_fp64"

        simd_nelements = "get_simd_size()/sizeof(double)"

        imm1 = testcase.random_fimmediate()
        avec = vg.new_vector("double",
                             simd_nelements,
                             imm1,
                             vt=vio_type.INPUT)
        imm2 = testcase.random_fimmediate()
        bvec = vg.new_vector("double",
                             simd_nelements,
                             imm2,
                             vt=vio_type.INPUT)

        avreg_idx = rt.reserve_any_reg(type_tag="vreg")
        avreg = testcase.gen.vreg(avreg_idx)
        bvreg_idx = rt.reserve_any_reg(type_tag="vreg")
        bvreg = testcase.gen.vreg(bvreg_idx)

        asmblock  = testcase.gen.isaquirks(rt, dt.DOUBLE)

        asmblock += testcase.gen.zero_vreg(avreg, dt.DOUBLE)
        asmblock += testcase.gen.zero_vreg(bvreg, dt.DOUBLE)

        aareg_idx = rt.reserve_any_reg(type_tag="greg")
        aareg = testcase.gen.greg(aareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(avec, aareg)
        asmblock += testcase.gen.load_vector(aareg, 0, avreg, dt.DOUBLE)

        bareg_idx = rt.reserve_any_reg(type_tag="greg")
        bareg = testcase.gen.greg(bareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(bvec, bareg)
        asmblock += testcase.gen.load_vector(bareg, 0, bvreg, dt.DOUBLE)
        rt.unuse_reg(type_tag="greg", idx=bareg_idx)

        asmblock += testcase.gen.fmul(avreg, bvreg, avreg, dt.DOUBLE, dt.DOUBLE, dt.DOUBLE)
        asmblock += testcase.gen.store_vector(aareg, 0, avreg, dt.DOUBLE)
        rt.unuse_reg(type_tag="greg", idx=aareg_idx)

        rt.unuse_reg(type_tag="vreg", idx=avreg_idx)
        rt.unuse_reg(type_tag="vreg", idx=bvreg_idx)

        extra_prepare  = f"std::vector<double> result({simd_nelements});\n"
        # adding up in python will actually result in failures
        extra_prepare += f"std::fill(result.begin(), result.end(), {imm1}*{imm2});\n"

        check_function = vec_fp64_close(f"check_{test_name}", identity)
        testcase.add_test(test_name, rt, vg,
                      asmblock, 
                      check_function,
                      f"check_{test_name}({avec}, result)",
                      extra_prepare=extra_prepare)
