from asmgen.asmblocks.noarch import asmgen,reg_tracker
from asmgen.asmblocks.noarch import asm_data_type as dt
from asmgen.asmblocks.avx_fma import fma128,fma256,avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071
from asmgen.cppgen.writers import write_test_asmblock_func
from asmgen.cppgen.writers import write_test_func_declaration
from asmgen.cppgen.declarations import vargen,vio_type
from asmgen.compilation.tools import compiler
from asmgen.compilation.compiler_presets import cross_archs,cross_cxx_flags
from .testcase import testcase

from .asm_test_generator import asm_test_generator

from parameterized import parameterized, parameterized_class

import logging
import os
import platform
import random
import shutil
import unittest
import sys

from subprocess import Popen, PIPE
from typing import Union

MIN_PYTHON = (3, 9)
if sys.version_info < MIN_PYTHON:
    sys.exit("Python %s.%s or later is required.\n" % MIN_PYTHON)

# Adding tests:
# Add test to one of the generators in tests/generators or create a new one
# and make asm_test_generator inherit from it
#
# TODO: more tests

#TODO: armclang++ is arm only, so avx/armclang and rvv/armclang combos are unnecessary
#NOTE: rvv071 will only work with special 0.7.1 EPI llvm, which will not work with RVV 1.0
#      So let's remove testing it
gens = [fma128(), fma256(), avx512(), neon(), sve(), rvv()]
gen_dicts : list[dict[str,Union[str,asmgen]]] = [{
    "name": gen.__class__.__name__,
    "gen": gen
    } for gen in gens]
cxx_dicts = [{
    "cxx_name" : name,
    } for name in ["g++", "clang++", "armclang++"]]
# NOTE: this needs python 3.9 or higher
parameters = [d1 | d2 for d1 in gen_dicts for d2 in cxx_dicts]

@parameterized_class(parameters)
class asm_correctness_test(unittest.TestCase,testcase):
    test_root = "test_run/asm_correct/"
    @classmethod
    def setUpClass(cls):
        cls.name : str = cls.name
        cls.cxx_name : str = cls.cxx_name
        cls.gen : asmgen = cls.gen
        # only used for existence check
        compiler_exe = shutil.which(cls.cxx_name)
        if None == compiler_exe:
            raise unittest.SkipTest(f"Compiler {cls.cxx_name} not found, skipping")

        cls.testlib_name = f"libtest_asm_correct_{cls.name}.so"
        cls.testlib_source  = f"// uarch_bench; Generator: {cls.name}\n"
        cls.testlib_source += "// generated C++/inline ASM code for ASM correctness tests\n"
        cls.header_source  = f"// uarch_bench; Generator: {cls.name}\n"
        cls.header_source += "// generated C++ header for ASM correctness tests\n"

        log = logging.getLogger("ASMCORTEST")

        cls.testlib_source += "#include <cstddef>\n"
        cls.testlib_source += "#include <cstdlib>\n"
        cls.testlib_source += "#include <cmath>\n"
        cls.testlib_source += "#include <limits>\n"
        # Add simd_size function so it's available for tests
        cls.testlib_source += cls.gen.c_simd_size_function

        # Generating test sources
        vg = vargen()
        rt = reg_tracker(cls.gen.max_gregs, cls.gen.max_vregs, cls.gen.max_fregs)

        # Run all generators
        generators = [getattr(asm_test_generator,name) for name in dir(asm_test_generator) if name.startswith("generate_")]
        [generator(cls,rt,vg) for generator in generators]

        # Easier to debug later
        clang_format_exe = shutil.which("clang-format")
        if not None == clang_format_exe:
            cmd = f"{clang_format_exe}"
            p = Popen([cmd,"--style=llvm"],stdin=PIPE, stdout=PIPE, stderr=PIPE)
            pretty_testlib_source = p.communicate(input=cls.testlib_source.encode())[0].decode()
            if 0 == p.returncode:
                cls.testlib_source = pretty_testlib_source
        if not None == clang_format_exe:
            cmd = f"{clang_format_exe}"
            p = Popen([cmd,"--style=llvm"],stdin=PIPE, stdout=PIPE, stderr=PIPE)
            pretty_header_source = p.communicate(input=cls.header_source.encode())[0].decode()
            if 0 == p.returncode:
                cls.header_source = pretty_header_source

        if not os.path.exists(cls.test_root):
            os.makedirs(cls.test_root)

        # save sources for later inspection
        source_path = os.path.join(cls.test_root, f"testlib_source_{cls.name}.cpp")
        header_path = os.path.join(cls.test_root, f"testlib_header_{cls.name}.hpp")

        source_file = open(source_path, mode="w")
        source_file.write(cls.testlib_source)
        source_file.close()

        header_file = open(header_path, mode="w")
        header_file.write(cls.header_source)
        header_file.close()

        cross_arch = cross_archs[cls.name]

        cls.compilation_failed = False
        if platform.machine() == cross_arch:
            cxx = compiler(cls.cxx_name,cls.name)
            result = cxx.compile_lib(cls.testlib_source, os.path.join(cls.test_root, cls.testlib_name))
            cls.compilation_failed = 0 == result
        else:
            log.debug(f"code arch: {cross_arch}, platform arch: {platform.machine()}")
            # Look up if cxx flags exist for compiler
            if not cls.cxx_name in cross_cxx_flags[cross_arch]:
                log.debug(f"skipping testlib compilation on generator {cls.name} because it's not supported on host and can't cross-compiler with {cls.cxx_name}")
                return
            log.debug(f"{cls.name} not supported on host, will cross compile and emulate")
            cxx = compiler(cls.cxx_name,cls.name)
            result = cxx.compile_lib(cls.testlib_source, os.path.join(cls.test_root, cls.testlib_name), cross_arch)
            cls.compilation_failed = 0 == result

    def setUp(self):
        if self.compilation_failed:
            self.fail(f"Compilation failed with {self.cxx_name}!")

    @staticmethod
    def random_immediate():
        # TODO: gen should have max_imm or something
        return random.randint(0,2**8)

    @staticmethod
    def random_fimmediate():
        return random.uniform(0.0,1.0)

    @classmethod
    def add_test(cls, name : str,
                 rt : reg_tracker,
                 vg : vargen,
                 asmblock : str,
                 check_function_definition : str,
                 check : str,
                 extra_prepare : str =""):

        header = vg.get_includes()
        preparation = vg.get_declarations()
        preparation += "\n"
        preparation += extra_prepare
        clobbered_gregs = [cls.gen.greg(i) for i in rt.get_clobbered_gregs()]
        clobbered_vregs = [cls.gen.vreg(i) for i in rt.get_clobbered_vregs()]

        inputs : list[tuple[str,str,str]] = \
                [(varname,"m",init) for varname,init,vt in vg.get_variables() \
                if vt == vio_type.INPUT]
        outputs : list[tuple[str,str,str]] = \
                [(varname,"=m",init) for varname,init,vt in vg.get_variables() \
                if vt == vio_type.OUTPUT]
        asmblock += cls.gen.operands(inputs, outputs, 
                                 clobbered_gregs+clobbered_vregs+["memory"])
        rt.reset()
        vg.reset_variables()

        tparams = {
            "function_name" : name,
            "function_params" : "",
            "header" : header,
            "prepare" : preparation,
            "checkfun_definition": check_function_definition,
            "check" : check,
            "analyze" : ""
        }
        cls.testlib_source += write_test_asmblock_func(name, asmblock, tparams)
        cls.header_source += write_test_func_declaration(name, tparams)

    @staticmethod
    def wrap_test_call(function_name):
        return f"""
        int main()
        {{
            return {function_name}();
        }}
        """


    #######################
    # Test execution here #
    #######################

#    @parameterized.expand([
#        ['add_imm_to_greg'],
#        ['add_imm_to_greg_2_times'],
#        ['add_greg_to_greg'],
#        ['mov_imm_to_greg'],
#        ['mov_fp32_to_vreg'],
#        ['mov_fp64_to_vreg'],
#        ['fp32_fma'],
#        ['fp64_fma'],
#        ]) 
    @parameterized.expand([
        [name.replace("generate_","")] for name in dir(asm_test_generator) if name.startswith("generate_")
        ]) 
    def test_asm(self, test_name):
        log = logging.getLogger("ASMCORTEST")
        source = self.header_source+"\n"+self.wrap_test_call(test_name)
        cxx = compiler(self.cxx_name,self.name)
        exe : str = ""
        exe_name = f"test_{test_name}_{self.name}"

        log.debug(f"testing {test_name} with {self.name}")

        cross_arch = cross_archs[self.name]
        cmd = []

        if platform.machine() == cross_arch:
            cxx.compile_exe(source, os.path.join(self.test_root,exe_name), [os.path.join(self.test_root,self.testlib_name)])

            exe = shutil.which(exe_name,path=self.test_root)
            if None == exe:
                self.fail(f"Compilation failed with generator {self.name} for {test_name}")
            cmd = [exe]
        else:
            if not self.cxx_name in cross_cxx_flags[cross_arch]:
                self.skipTest(f"Skipping: Can't cross-compile for {self.name} with {self.cxx_name}")
            cxx.compile_exe(source, os.path.join(self.test_root,exe_name), [os.path.join(self.test_root,self.testlib_name)], cross_arch)

            exe = shutil.which(exe_name,path=self.test_root)
            if None == exe:
                self.fail(f"Compilation failed with generator {self.name} for {test_name}")
        if not self.gen.supported_on_host():
            #TODO: very OS specific, needs some logic
            emulation_commands = {
                    "rvv071" : ["qemu-riscv64-static", "-L", "/usr/riscv64-linux-gnu",
                             "-cpu", "rv64,v=on,zba=on,vlen=512,vext_spec=v0.7.1"],
                    "rvv" : ["qemu-riscv64-static", "-L", "/usr/riscv64-linux-gnu",
                             "-cpu", "rv64,v=on,zba=on,vlen=512,vext_spec=v1.0"],
                    "neon" : ["qemu-aarch64-static", "-L", "/usr/aarch64-linux-gnu",
                              "-E","LD_LIBRARY_PATH=/usr/aarch64-linux-gnu/lib64"],
                    "sve" : ["qemu-aarch64-static", "-L", "/usr/aarch64-linux-gnu",
                              "-E","LD_LIBRARY_PATH=/usr/aarch64-linux-gnu/lib64",
                             "-cpu", "max,sve=on,sve512=on"],
                    "fma128" : ["sde", "-future", "--"],
                    "fma256" : ["sde", "-future", "--"],
                    "avx512" : ["sde", "-future", "--"],
                    }
            cmd = emulation_commands[self.name]+[exe]
        log.debug(f"Running test {test_name} with: {' '.join(cmd)}")
        p = Popen(cmd,stdin=PIPE,stdout=PIPE,stderr=PIPE)
        process_out = p.communicate()
        output = process_out[0].decode()
        errout = process_out[1].decode()
        log.debug(f"Execution stdout: {output}")
        log.debug(f"Execution stderr: {errout}")
        self.assertEqual(0, p.returncode)


        

def main():
    # useful for debugging tests
    logging.basicConfig(level=logging.DEBUG, format='%(name)s %(levelname)s %(message)s')
    logging.getLogger("ASMCORTEST").setLevel(logging.DEBUG)
    logging.getLogger("COMPILATION").setLevel(logging.DEBUG)
    unittest.main()

if "__main__" == __name__:
    main()
