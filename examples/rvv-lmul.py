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
from asmgen.registers import asm_data_type as adt, reg_tracker
from asmgen.asmblocks.operations import modifier as mod


def main():
    gen = rvv()


    gen.set_output_inline(yesno=False)

    for lmul in [1,2,4,8]:
        gen.set_parameter("LMUL", lmul)
        rt = reg_tracker(reg_type_init_list=[
            ("greg",gen.max_gregs),
            ("freg",gen.max_fregs),
            ("vreg",gen.max_vregs),
            ("treg",gen.max_tregs(dt=adt.FP64)),
            ])

        aidx = rt.reserve_any_reg("vreg")
        a = gen.vreg(aidx)
        bidx = rt.reserve_any_reg("vreg")
        b = gen.vreg(bidx)
        cidx = rt.reserve_any_reg("vreg")
        c = gen.vreg(cidx)

        bfidx = rt.reserve_any_reg("freg")
        bf = gen.freg(bfidx,dt=adt.FP64)


        asmblock = gen.isaquirks(dt=adt.FP64,rt=rt)

        asmblock += gen.fma(
                adreg=a,
                bdreg=b,
                cdreg=c,
                a_dt=adt.FP64,
                b_dt=adt.FP64,
                c_dt=adt.FP64)
        asmblock += gen.fma(
                adreg=a,
                bdreg=bf,
                cdreg=c,
                a_dt=adt.FP64,
                b_dt=adt.FP64,
                c_dt=adt.FP64,
                modifiers={mod.VF})

        asmblock += gen.isaendquirks(dt=adt.FP64, rt=rt)

        rt.unuse_reg("vreg", aidx)
        rt.unuse_reg("vreg", bidx)
        rt.unuse_reg("vreg", cidx)
        rt.unuse_reg("freg", bfidx)

        print(asmblock)


if __name__ == "__main__":
    main()
