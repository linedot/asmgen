# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

# expected output:
#
# .section .text
# .global myfunction
# myfunction:
#   # a0 = x
# # a1 = y
# # f10 = alpha
# # a2 = n
# vsetvli t0, zero, e64, m1, ta, ma
# slli t0,t0,3
# .loop:
# vle64.v v0, (a0)
# vle64.v v1, (a1)
# vfmacc.vf v1,f10,v0
# vse64.v v1, (a1)
# add a0,a0,t0
# add a1,a1,t0
# add a2,a2,-1
# bne a2,zero,.loop


from asmgen.registers import (
        adt_size,
        asm_data_type as adt,
        reg_tracker
)
from asmgen.asmblocks.noarch import asmgen
from asmgen.callconv.callconv import callconv
from asmgen.callconv.fngen import fngen
from asmgen.asmblocks.noarch import comparison
from asmgen.asmblocks.operations import modifier as mod

import sys
import importlib

isa_modules = {
        "fma128" : "avx_fma",
        "fma256" : "avx_fma",
        "avx512" : "avx_fma",
        }

def main():

    dt = adt.FP64

    isa = 'rvv'
    if len(sys.argv) == 2:
        isa = sys.argv[1]

    module_name = isa
    if isa in isa_modules:
        module_name = isa_modules[isa]
    generator_module = importlib.import_module(f"asmgen.asmblocks.{module_name}")

    generator_class = getattr(generator_module, isa)

    gen : asmgen = generator_class()
    gen.set_output_inline(yesno=False)
    rt = reg_tracker(reg_type_init_list=[
        ("greg",gen.max_gregs),
        ("freg",gen.max_fregs),
        ("vreg",gen.max_vregs),
        ("treg",gen.max_tregs(dt=dt)),
        ])
    cc = gen.create_callconv()

    func = fngen(gen=gen, rt=rt)

    cc.add_param("greg", "x")
    cc.add_param("greg", "y")
    cc.add_param("freg", "alpha", adt.FP64)
    cc.add_param("greg", "n")


    fnname = "myfunction"

    func.init_cc(cc=cc)



    addr_x_idx     = rt.aliased_regs["greg"]["x"]
    addr_y_idx     = rt.aliased_regs["greg"]["y"]
    #addr_alpha_idx = rt.aliased_regs["greg"]["alpha"]
    n_idx = rt.aliased_regs["greg"]["n"]


    asmheader = (
             ".section .text\n"
            f".global {fnname}\n"
            f"{fnname}:\n  "
        )

    innerblock = ""

    addr_x = gen.greg(addr_x_idx)
    addr_y = gen.greg(addr_y_idx)
    #addr_alpha = gen.greg(addr_alpha_idx)
    n = gen.greg(n_idx)


    x_idx = rt.reserve_any_reg("vreg")
    x = gen.vreg(x_idx)
    y_idx = rt.reserve_any_reg("vreg")
    y = gen.vreg(y_idx)
    alpha_idx = rt.aliased_regs["freg"]["alpha"]
    alpha = gen.freg(alpha_idx, dt=dt)

    innerblock += gen.isaquirks(dt=dt,rt=rt)

    if "vlen" in rt.aliased_regs["greg"]:
        vlenidx = rt.aliased_regs["greg"]["vlen"]
        vlen = gen.greg(vlenidx)

        innerblock += gen.shift_greg_left(
                reg=vlen,
                bit_count=adt_size(dt).bit_length()-1)
    
    can_vf = True
    fma_mods = {mod.VF}
    breg = alpha
    try:
        gen.fma(
            adreg=gen.vreg(0),
            bdreg=gen.freg(1, adt.FP64),
            cdreg=gen.vreg(2),
            a_dt=dt,
            b_dt=dt,
            c_dt=dt,
            modifiers=fma_mods
            )
    except:
        can_vf = False
        fma_mods = {}

    if not can_vf:
        alpha_vreg_idx = rt.reserve_any_reg("vreg")
        alpha_vreg = gen.vreg(alpha_vreg_idx)
        innerblock += gen.fill_vector(sreg=alpha, vreg=alpha_vreg, dt=dt)
        breg = alpha_vreg


#    innerblock += gen.load_scalar_immoff(areg=addr_alpha,
#                                         offset=0,
#                                         freg=alpha,
#                                         dt=dt)

    innerblock += gen.label(label="loop")

    innerblock += gen.load_vector(areg=addr_x,
                                  vreg=x,
                                  dt=dt)
    innerblock += gen.load_vector(areg=addr_y,
                                  vreg=y,
                                  dt=dt)

    innerblock += gen.fma(
            adreg=x,
            bdreg=breg,
            cdreg=y,
            a_dt=dt,
            b_dt=dt,
            c_dt=dt,
            modifiers=fma_mods)

    innerblock += gen.store_vector(areg=addr_y,
                                   vreg=y,
                                   dt=dt)

    if "vlen" in rt.aliased_regs["greg"]:
        innerblock += gen.add_greg_greg(dst=addr_x,
                                        reg1=addr_x,
                                        reg2=vlen)
        innerblock += gen.add_greg_greg(dst=addr_y,
                                        reg1=addr_y,
                                        reg2=vlen)
    else:
        innerblock += gen.add_greg_voff(reg=addr_x, offset=1, dt=dt)
        innerblock += gen.add_greg_voff(reg=addr_y, offset=1, dt=dt)

    innerblock += gen.add_greg_imm(reg=n,imm=-1)

    innerblock += gen.cb(reg1=n, reg2=None, cmp=comparison.NZ, label="loop")


    fnsave,fnload,fnrestore = func.get_boilerplate(cc=cc)

    asmblock = asmheader + fnsave + fnload + innerblock + fnrestore


    print(asmblock)



if __name__ == "__main__":
    main()
