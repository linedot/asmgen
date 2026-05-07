from asmgen.asmblocks.rvv_opdna1.rvv_opdna1_base import rvv_opdna1
load = rvv_opdna1(lmul_getter=lambda : 1)
load.inst_base = "vl"
from asmgen.asmblocks.types.rvv_types import rvv_vreg
from asmgen.asmblocks.types.riscv64_types import riscv64_greg
vs = [rvv_vreg(i) for i in range(4)]
vid = rvv_vreg(5)
a = riscv64_greg(0)
s = riscv64_greg(1)
from asmgen.asmblocks.operations import opdna1_modifier as mod
from asmgen.registers import asm_data_type as adt
print(load(dregs=vs[:1], areg=a, dt=adt.FP64, modifiers={}))
print(load(dregs=vs[:1], areg=a, streg=s, dt=adt.FP64, modifiers={mod.GSTRIDE}))
print(load(dregs=vs[:2], areg=a, nstructs=2, dt=adt.FP64, modifiers={mod.STRUCT}))
print(load(dregs=vs[:1], areg=a, vidxreg=vid, dt=adt.FP64, modifiers={mod.VINDEX}))
print(load(dregs=vs[:2], areg=a, streg=s, nstructs=2, dt=adt.FP64, modifiers={mod.STRUCT,mod.GSTRIDE}))
print(load(dregs=vs[:2], areg=a, vidxreg=vid, nstructs=2, dt=adt.FP64, modifiers={mod.STRUCT,mod.VINDEX}))
