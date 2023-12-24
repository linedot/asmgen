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
        # We need to vsetvli for fp32
        if "rvv" == testcase.name or "rvv071" == testcase.name:
            tmpreg_idx = rt.reserve_any_greg()
            tmpreg = testcase.gen.greg(tmpreg_idx)
            asmblock += testcase.gen.vsetvlmax(tmpreg, dt.SINGLE)
            rt.unuse_greg(tmpreg_idx)

        areg_idx = rt.reserve_any_greg()
        areg = testcase.gen.greg(areg_idx)
        asmblock += testcase.gen.zero_greg(areg)
        asmblock += testcase.gen.mov_param_to_greg(varname, areg)

        vreg_idx = rt.reserve_any_vreg()
        vreg = testcase.gen.vreg(vreg_idx)
        asmblock += testcase.gen.zero_vreg(vreg, dt.SINGLE)
        asmblock += testcase.gen.load_vector(areg, 0, vreg, dt.SINGLE)

        rt.unuse_greg(areg_idx)
        rt.unuse_vreg(vreg_idx)

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
        # We need to vsetvli for fp64
        if "rvv" == testcase.name or "rvv071" == testcase.name:
            tmpreg_idx = rt.reserve_any_greg()
            tmpreg = testcase.gen.greg(tmpreg_idx)
            asmblock += testcase.gen.vsetvlmax(tmpreg, dt.DOUBLE)
            rt.unuse_greg(tmpreg_idx)

        areg_idx = rt.reserve_any_greg()
        areg = testcase.gen.greg(areg_idx)
        asmblock += testcase.gen.zero_greg(areg)
        asmblock += testcase.gen.mov_param_to_greg(varname, areg)

        vreg_idx = rt.reserve_any_vreg()
        vreg = testcase.gen.vreg(vreg_idx)
        asmblock += testcase.gen.zero_vreg(vreg, dt.DOUBLE)
        asmblock += testcase.gen.load_vector(areg, 0, vreg, dt.DOUBLE)

        rt.unuse_greg(areg_idx)
        rt.unuse_vreg(vreg_idx)

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

        avreg_idx = rt.reserve_any_vreg()
        avreg = testcase.gen.vreg(avreg_idx)
        bvreg_idx = rt.reserve_any_vreg()
        bvreg = testcase.gen.vreg(bvreg_idx)
        cvreg_idx = rt.reserve_any_vreg()
        cvreg = testcase.gen.vreg(cvreg_idx)

        asmblock  = testcase.gen.zero_vreg(avreg, dt.SINGLE)
        asmblock += testcase.gen.zero_vreg(bvreg, dt.SINGLE)
        asmblock += testcase.gen.zero_vreg(cvreg, dt.SINGLE)

        # We need a true p0, otherwise UB
        if "sve" == testcase.name:
            asmblock += testcase.gen.ptrue(testcase.gen.preg(0), dt.SINGLE)

        # We need to vsetvli for fp64
        if "rvv" == testcase.name or "rvv071" == testcase.name:
            tmpreg_idx = rt.reserve_any_greg()
            tmpreg = testcase.gen.greg(tmpreg_idx)
            asmblock += testcase.gen.vsetvlmax(tmpreg, dt.SINGLE)
            rt.unuse_greg(tmpreg_idx)

        aareg_idx = rt.reserve_any_greg()
        aareg = testcase.gen.greg(aareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(avec, aareg)
        asmblock += testcase.gen.load_vector(aareg, 0, avreg, dt.SINGLE)
        rt.unuse_greg(aareg_idx)

        bareg_idx = rt.reserve_any_greg()
        bareg = testcase.gen.greg(bareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(bvec, bareg)
        asmblock += testcase.gen.load_vector(bareg, 0, bvreg, dt.SINGLE)
        rt.unuse_greg(bareg_idx)

        careg_idx = rt.reserve_any_greg()
        careg = testcase.gen.greg(careg_idx)
        asmblock += testcase.gen.mov_param_to_greg(cvec, careg)
        asmblock += testcase.gen.load_vector(careg, 0, cvreg, dt.SINGLE)
        asmblock += testcase.gen.fma(avreg, bvreg, cvreg, dt.SINGLE)
        asmblock += testcase.gen.store_vector(careg, 0, cvreg, dt.SINGLE)
        rt.unuse_greg(careg_idx)

        rt.unuse_vreg(avreg_idx)
        rt.unuse_vreg(bvreg_idx)
        rt.unuse_vreg(cvreg_idx)

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

        avreg_idx = rt.reserve_any_vreg()
        avreg = testcase.gen.vreg(avreg_idx)
        bvreg_idx = rt.reserve_any_vreg()
        bvreg = testcase.gen.vreg(bvreg_idx)
        cvreg_idx = rt.reserve_any_vreg()
        cvreg = testcase.gen.vreg(cvreg_idx)

        asmblock  = testcase.gen.zero_vreg(avreg, dt.DOUBLE)
        asmblock += testcase.gen.zero_vreg(bvreg, dt.DOUBLE)
        asmblock += testcase.gen.zero_vreg(cvreg, dt.DOUBLE)

        # We need a true p0, otherwise UB
        if "sve" == testcase.name:
            asmblock += testcase.gen.ptrue(testcase.gen.preg(0), dt.DOUBLE)

        # We need to vsetvli for fp64
        if "rvv" == testcase.name or "rvv071" == testcase.name:
            tmpreg_idx = rt.reserve_any_greg()
            tmpreg = testcase.gen.greg(tmpreg_idx)
            asmblock += testcase.gen.vsetvlmax(tmpreg, dt.DOUBLE)
            rt.unuse_greg(tmpreg_idx)

        aareg_idx = rt.reserve_any_greg()
        aareg = testcase.gen.greg(aareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(avec, aareg)
        asmblock += testcase.gen.load_vector(aareg, 0, avreg, dt.DOUBLE)
        rt.unuse_greg(aareg_idx)

        bareg_idx = rt.reserve_any_greg()
        bareg = testcase.gen.greg(bareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(bvec, bareg)
        asmblock += testcase.gen.load_vector(bareg, 0, bvreg, dt.DOUBLE)
        rt.unuse_greg(bareg_idx)

        careg_idx = rt.reserve_any_greg()
        careg = testcase.gen.greg(careg_idx)
        asmblock += testcase.gen.mov_param_to_greg(cvec, careg)
        asmblock += testcase.gen.load_vector(careg, 0, cvreg, dt.DOUBLE)
        asmblock += testcase.gen.fma(avreg, bvreg, cvreg, dt.DOUBLE)
        asmblock += testcase.gen.store_vector(careg, 0, cvreg, dt.DOUBLE)
        rt.unuse_greg(careg_idx)

        rt.unuse_vreg(avreg_idx)
        rt.unuse_vreg(bvreg_idx)
        rt.unuse_vreg(cvreg_idx)

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

        avreg_idx = rt.reserve_any_vreg()
        avreg = testcase.gen.vreg(avreg_idx)
        bvreg_idx = rt.reserve_any_vreg()
        bvreg = testcase.gen.vreg(bvreg_idx)

        asmblock  = testcase.gen.zero_vreg(avreg, dt.SINGLE)
        asmblock += testcase.gen.zero_vreg(bvreg, dt.SINGLE)

        # We need a true p0, otherwise UB
        if "sve" == testcase.name:
            asmblock += testcase.gen.ptrue(testcase.gen.preg(0), dt.SINGLE)

        # We need to vsetvli for fp64
        if "rvv" == testcase.name or "rvv071" == testcase.name:
            tmpreg_idx = rt.reserve_any_greg()
            tmpreg = testcase.gen.greg(tmpreg_idx)
            asmblock += testcase.gen.vsetvlmax(tmpreg, dt.SINGLE)
            rt.unuse_greg(tmpreg_idx)

        aareg_idx = rt.reserve_any_greg()
        aareg = testcase.gen.greg(aareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(avec, aareg)
        asmblock += testcase.gen.load_vector(aareg, 0, avreg, dt.SINGLE)

        bareg_idx = rt.reserve_any_greg()
        bareg = testcase.gen.greg(bareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(bvec, bareg)
        asmblock += testcase.gen.load_vector(bareg, 0, bvreg, dt.SINGLE)
        rt.unuse_greg(bareg_idx)

        asmblock += testcase.gen.fmul(avreg, bvreg, avreg, dt.SINGLE)
        asmblock += testcase.gen.store_vector(aareg, 0, avreg, dt.SINGLE)
        rt.unuse_greg(aareg_idx)

        rt.unuse_vreg(avreg_idx)
        rt.unuse_vreg(bvreg_idx)

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

        avreg_idx = rt.reserve_any_vreg()
        avreg = testcase.gen.vreg(avreg_idx)
        bvreg_idx = rt.reserve_any_vreg()
        bvreg = testcase.gen.vreg(bvreg_idx)

        asmblock  = testcase.gen.zero_vreg(avreg, dt.DOUBLE)
        asmblock += testcase.gen.zero_vreg(bvreg, dt.DOUBLE)

        # We need a true p0, otherwise UB
        if "sve" == testcase.name:
            asmblock += testcase.gen.ptrue(testcase.gen.preg(0), dt.DOUBLE)

        # We need to vsetvli for fp64
        if "rvv" == testcase.name or "rvv071" == testcase.name:
            tmpreg_idx = rt.reserve_any_greg()
            tmpreg = testcase.gen.greg(tmpreg_idx)
            asmblock += testcase.gen.vsetvlmax(tmpreg, dt.DOUBLE)
            rt.unuse_greg(tmpreg_idx)

        aareg_idx = rt.reserve_any_greg()
        aareg = testcase.gen.greg(aareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(avec, aareg)
        asmblock += testcase.gen.load_vector(aareg, 0, avreg, dt.DOUBLE)

        bareg_idx = rt.reserve_any_greg()
        bareg = testcase.gen.greg(bareg_idx)
        asmblock += testcase.gen.mov_param_to_greg(bvec, bareg)
        asmblock += testcase.gen.load_vector(bareg, 0, bvreg, dt.DOUBLE)
        rt.unuse_greg(bareg_idx)

        asmblock += testcase.gen.fmul(avreg, bvreg, avreg, dt.DOUBLE)
        asmblock += testcase.gen.store_vector(aareg, 0, avreg, dt.DOUBLE)
        rt.unuse_greg(aareg_idx)

        rt.unuse_vreg(avreg_idx)
        rt.unuse_vreg(bvreg_idx)

        extra_prepare  = f"std::vector<double> result({simd_nelements});\n"
        # adding up in python will actually result in failures
        extra_prepare += f"std::fill(result.begin(), result.end(), {imm1}*{imm2});\n"

        check_function = vec_fp64_close(f"check_{test_name}", identity)
        testcase.add_test(test_name, rt, vg,
                      asmblock, 
                      check_function,
                      f"check_{test_name}({avec}, result)",
                      extra_prepare=extra_prepare)
