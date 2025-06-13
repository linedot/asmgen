"""
Test the Calling convention abstraction
"""

import unittest


from parameterized import parameterized_class
from mako.template import Template

from asmgen.callconv.callconv import callconv
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.avx_fma import fma128,fma256,avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071
from asmgen.asmblocks.sve import sve

# base ISA classes
from asmgen.asmblocks.riscv64 import riscv64
from asmgen.asmblocks.aarch64 import aarch64
from asmgen.asmblocks.avx_fma import avxbase

@parameterized_class([
    {"name": "fma128", "gen": fma128()},
    {"name": "fma256", "gen": fma256()},
    {"name": "avx512", "gen": avx512()},
    {"name": "neon", "gen": neon()},
    {"name": "sve", "gen": sve()},
    {"name": "rvv", "gen": rvv()},
    {"name": "rvv071", "gen": rvv071()},
])
class test_callconv_gemmukr(unittest.TestCase):

    def setUp(self):
        self.cc = self.gen.create_callconv()


        self.cc.add_param('greg', "m")
        self.cc.add_param('greg', "n")
        self.cc.add_param('greg', "k")
        self.cc.add_param('greg', "alpha")
        self.cc.add_param('greg', "a")
        self.cc.add_param('greg', "b")
        self.cc.add_param('greg', "beta")
        self.cc.add_param('greg', "c")
        self.cc.add_param('greg', "rs_c")
        self.cc.add_param('greg', "cs_c")
        self.cc.add_param('greg', "data")
        self.cc.add_param('greg', "cntx")


        self.gregs_used = []
        self.fregs_used = []

        for name,(tag,idx) in self.cc.get_params().items():

            location = ''
            if 'greg' == tag:
                self.gregs_used.append(idx)
            elif 'freg' == tag:
                self.fregs_used.append(idx)


    expected_save_blocks = [
        (riscv64,(
            "add sp,sp,-64\n"
            "sd a0,0(sp)\n"
            "sd a1,8(sp)\n"
            "sd a2,16(sp)\n"
            "sd a3,24(sp)\n"
            "sd a4,32(sp)\n"
            "sd a5,40(sp)\n"
            "sd a6,48(sp)\n"
            "sd a7,56(sp)\n"
        )),
        (aarch64, (
            "add x31,x31,#-64\n"
            "str x0,[x31,#0]\n"
            "str x1,[x31,#8]\n"
            "str x2,[x31,#16]\n"
            "str x3,[x31,#24]\n"
            "str x4,[x31,#32]\n"
            "str x5,[x31,#40]\n"
            "str x6,[x31,#48]\n"
            "str x7,[x31,#56]\n")
         ),
        (avxbase,(
            "addq $-48,%%rsp\n"
            "movq %%r8,(%%rsp)\n"
            "movq %%r9,8(%%rsp)\n"
            "movq %%rcx,16(%%rsp)\n"
            "movq %%rdx,24(%%rsp)\n"
            "movq %%rsi,32(%%rsp)\n"
            "movq %%rdi,40(%%rsp)\n")
         )
    ]

    expected_restore_blocks = [
        (riscv64,(
            "ld a0,0(sp)\n"
            "ld a1,8(sp)\n"
            "ld a2,16(sp)\n"
            "ld a3,24(sp)\n"
            "ld a4,32(sp)\n"
            "ld a5,40(sp)\n"
            "ld a6,48(sp)\n"
            "ld a7,56(sp)\n"
            "add sp,sp,64\n")
         ),
        (aarch64, (
            "ldr x0,[x31,#0]\n"
            "ldr x1,[x31,#8]\n"
            "ldr x2,[x31,#16]\n"
            "ldr x3,[x31,#24]\n"
            "ldr x4,[x31,#32]\n"
            "ldr x5,[x31,#40]\n"
            "ldr x6,[x31,#48]\n"
            "ldr x7,[x31,#56]\n"
            "add x31,x31,#64\n")
         ),
        (avxbase,(
            "movq (%%rsp),%%r8\n"
            "movq 8(%%rsp),%%r9\n"
            "movq 16(%%rsp),%%rcx\n"
            "movq 24(%%rsp),%%rdx\n"
            "movq 32(%%rsp),%%rsi\n"
            "movq 40(%%rsp),%%rdi\n"
            "addq $48,%%rsp\n")
         )
    ]

    def test_save_restore(self):

        expected_save_block = "UNKNOWN"
        for base,block in self.expected_save_blocks:
            if isinstance(self.gen,base):
                expected_save_block = block

        expected_restore_block = "UNKNOWN"
        for base,block in self.expected_restore_blocks:
            if isinstance(self.gen,base):
                expected_restore_block = block

        self.gen.set_output_inline(yesno=False)

        save_block = self.cc.save_before_call(
                gen=self.gen,
                regs={
                    'greg': self.gregs_used,
                    'freg': self.fregs_used,
                })
        restore_block = self.cc.restore_after_call(
                gen=self.gen,
                regs={
                    'greg': self.gregs_used,
                    'freg': self.fregs_used,
                })


        #print(save_block)
        #print(restore_block)

        self.assertEqual(expected_save_block, save_block)
        self.assertEqual(expected_restore_block, restore_block)
        #gregs_used = []

        #for name,(tag,idx) in cc.get_params().items():

        #    location = ''
        #    if 'greg' == tag:
        #        location = gen.greg(idx)
        #        gregs_used.append(idx)
        #    elif 'freg' == tag:
        #        location = gen.freg(idx, adt.FP64)
        #    elif 'sp' == tag:
        #        location = f"{idx}(sp)"


        #    print(f"{location} <- {name}")

        #gregs_to_save = set(gregs_used).intersection(set(cc.greg_caller_save_list))

        #gregs_to_save = [str(gen.greg(idx)) for idx in gregs_to_save]

        #asm_tpl = Template("""
        #.section .rodata
        #paramout: .asciz "Parameter %s = %llu\\n"
        #% for i,name in enumerate(params.keys()):
        #  paramname${i}: .asciz "${name}"
        #% endfor

        #<% sp_offset = 8*(len(gregs_to_save)+1)  %>

        #.section .text
        #.global gemm_kernel
        #gemm_kernel:
        #addi sp, sp, -${sp_offset}
        #sd ra, (sp)

        #% for i,greg in enumerate(gregs_to_save):
        #    sd ${greg}, ${8*(i+1)}(sp)
        #% endfor
        #% for i,name in enumerate(params.keys()):
        #    <%
        #        location = ''
        #        tag,idx = params[name]
        #        if 'greg' == tag:
        #            location = str(gen.greg(idx))
        #        elif 'freg' == tag:
        #            location = str(gen.freg(idx))
        #        elif 'sp' == tag:
        #            location = f"{idx+sp_offset}(sp)"

        #        if location in gregs_to_save:
        #            idx = gregs_to_save.index(location)
        #            location = f"{8*(idx+1)}(sp)"
        #    %>
        #    % if "sp" in str(location):
        #    ld a2, ${location}
        #    % else:
        #    mv a2, ${location}
        #    % endif
        #    la a1, paramname${i}
        #    la a0, paramout
        #    call printf
        #% endfor
        #ld ra, (sp)
        #addi sp, sp, ${sp_offset}
        #ret
        #""")

        #asm = asm_tpl.render(params=cc.get_params(),
        #                     gen=gen,
        #                     gregs_to_save=gregs_to_save)

        #cpp_code = """
        ##include <cstdint>
        #extern "C" void gemm_kernel(std::uint64_t m, std::uint64_t n, std::uint64_t k,
        #    void* alpha, void* a, void* b,
        #    void* beta, void* c,
        #    int rs_c, int cs_c,
        #    void* data, void* cntx);
        #int main()
        #{
        #    gemm_kernel(100, 101, 102,
        #        (void*)0x1, (void*)0x2, (void*)0x3, (void*)0x4, (void*)0x5,
        #        10, 14,
        #        (void*)0x6, (void*)0x7);
        #}
        #"""

        #with open("test.cpp","w+") as f:
        #    f.write(cpp_code)

        #with open("test.s","w+") as f:
        #    f.write(asm)

