# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

# Output should be:
#
# vsetvli t0, zero, e64, m1, ta, ma
# vfmacc.vv v2,v1,v0
# vfmacc.vf v2,f0,v0
# 
# vsetvli t0, zero, e64, m2, ta, ma
# vfmacc.vv v4,v2,v0
# vfmacc.vf v4,f0,v0
# 
# vsetvli t0, zero, e64, m4, ta, ma
# vfmacc.vv v8,v4,v0
# vfmacc.vf v8,f0,v0
# 
# vsetvli t0, zero, e64, m8, ta, ma
# vfmacc.vv v16,v8,v0
# vfmacc.vf v16,f0,v0


from asmgen.asmblocks.rvv import rvv
from asmgen.registers import (
        adt_size,
        asm_data_type as adt,
        reg_tracker
)
from asmgen.asmblocks.operations import modifier as mod


def main():
    gen = rvv()


    gen.set_output_inline(yesno=False)

    dt = adt.FP64

    for lmul in [1,2,4,8]:
        gen.set_parameter("LMUL", lmul)
        rt = reg_tracker(reg_type_init_list=[
            ("greg",gen.max_gregs),
            ("freg",gen.max_fregs),
            ("vreg",gen.max_vregs),
            ("treg",gen.max_tregs(dt=dt)),
            ])

        aidx = rt.reserve_any_reg("vreg")
        a = gen.vreg(aidx)
        bidx = rt.reserve_any_reg("vreg")
        b = gen.vreg(bidx)
        cidx = rt.reserve_any_reg("vreg")
        c = gen.vreg(cidx)

        aaddr_idx = rt.reserve_any_reg("greg")
        aaddr = gen.greg(aaddr_idx)
        baddr_idx = rt.reserve_any_reg("greg")
        baddr = gen.greg(baddr_idx)
        caddr_idx = rt.reserve_any_reg("greg")
        caddr = gen.greg(caddr_idx)

        bfidx = rt.reserve_any_reg("freg")
        bf = gen.freg(bfidx,dt=dt)

        bfaddr_idx = rt.reserve_any_reg("greg")
        bfaddr = gen.greg(bfaddr_idx)


        asmblock = gen.isaquirks(dt=dt,rt=rt)

        vlenidx = rt.aliased_regs["greg"]["vlen"]
        vlen = gen.greg(vlenidx)

        asmblock += gen.shift_greg_left(
                reg=vlen,
                bit_count=adt_size(dt).bit_length()-1)


        asmblock += gen.load_vector(
                areg=aaddr,
                vreg=a,
                dt=dt)
        asmblock += gen.load_vector(
                areg=baddr,
                vreg=b,
                dt=dt)
        asmblock += gen.load_vector(
                areg=caddr,
                vreg=c,
                dt=dt)

        asmblock += gen.fma(
                adreg=a,
                bdreg=b,
                cdreg=c,
                a_dt=dt,
                b_dt=dt,
                c_dt=dt)
        asmblock += gen.fma(
                adreg=a,
                bdreg=bf,
                cdreg=c,
                a_dt=dt,
                b_dt=dt,
                c_dt=dt,
                modifiers={mod.VF})


        asmblock += gen.add_greg_greg(dst=aaddr,
                                      reg1=aaddr,
                                      reg2=vlen)
        asmblock += gen.add_greg_greg(dst=baddr,
                                      reg1=baddr,
                                      reg2=vlen)
        asmblock += gen.add_greg_greg(dst=caddr,
                                      reg1=caddr,
                                      reg2=vlen)

        asmblock += gen.add_greg_imm(reg=bfaddr,
                                     imm=adt_size(dt))

        asmblock += gen.isaendquirks(dt=dt, rt=rt)

        rt.unuse_reg("vreg", aidx)
        rt.unuse_reg("vreg", bidx)
        rt.unuse_reg("vreg", cidx)
        rt.unuse_reg("freg", bfidx)

        print(asmblock)


if __name__ == "__main__":
    main()
