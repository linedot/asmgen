from ..registers import (
    reg_tracker,
    asm_data_type as adt,
    adt_triple,
    adt_size,
    asm_index_type as ait,
    data_reg,
    treg,vreg,freg,greg
)
from .riscv64 import riscv64
from .types.rvv_types import rvv_vreg

from .rvv_opd3 import rvv_fma,rvv_fmul

from typing import TypeAlias

class rvv(riscv64):

    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = rvv_vreg
    treg_type : TypeAlias = treg

    dt_suffixes = {
            adt.DOUBLE : "e64",
            adt.SINGLE : "e32",
            adt.HALF   : "e16",
            }
    it_suffixes = {
            ait.INT64 : "ei64",
            ait.INT32 : "ei32",
            ait.INT16 : "ei16",
            ait.INT8  : "ei8",
            }

    def __init__(self):
        super(rvv, self).__init__()
        self.fma = rvv_fma(asmwrap=self.asmwrap)
        self.fmul = rvv_fmul(asmwrap=self.asmwrap)

    def supportedby_cpuinfo(self, cpuinfo : str) -> bool:
         isa_idx = cpuinfo.find("rv64")
         if -1 == isa_idx:
             return False
         isa_idx = isa_idx+4
         extensions = cpuinfo[isa_idx:].split()[0]
         print(f"Extensions: {extensions}")
         return "v" in extensions

    def isaquirks(self, rt : reg_tracker, dt : adt):
        tmpreg_idx = rt.reserve_any_reg("greg")
        tmpreg = self.greg(tmpreg_idx)
        asmblock = self.vsetvlmax(tmpreg, dt)
        rt.unuse_reg("greg", tmpreg_idx)
        return asmblock

    def vreg(self, reg_idx : int) -> vreg_type:
        return rvv_vreg(reg_idx)

    def jvzero(self, vreg1 : vreg_type, freg : freg_type,
               vreg2 : vreg_type, greg : greg_type, label : str,
               dt : adt) -> str:
        dt_suf = self.fdt_suffixes[dt]
        asmblock  = self.asmwrap(f"fmv.{dt_suf}.x {freg},zero")
        # vec filled with 1 where element not-zero
        asmblock += self.asmwrap(f"vmfne.vf {vreg2},{vreg1},{freg}")
        # greg has number of elements that are not zero
        asmblock += self.asmwrap(f"vcpop.m {greg},{vreg2}")
        # if non-zero number of elements are non-zero, i.e greg has non-zero number,
        # we do not have zero, so we don't jump
        # So we jump when zero
        asmblock += self.asmwrap(f"beqz {greg},{label}")
        return asmblock

    @property
    def is_vla(self):
        return True

    def indexable_elements(self, dt : adt):
        return self.simd_size//adt_size(dt)

    @property
    def max_vregs(self):
        return 32

    @property
    def simd_size(self):
        return 1

    def simd_size_to_greg(self, reg : greg_type,
                          dt : adt):
        return self.asmwrap(f"csrr {reg}, vlenb")

    @property
    def c_simd_size_function(self):
        result  = "size_t get_simd_size() {\n"
        result += "    size_t byte_size = 0;\n"
        result += "    __asm__ volatile(\n"
        result += "        "+self.asmwrap("vsetvli %[byte_size], zero, e8, m1, ta, ma")
        result += "    : [byte_size] \"=r\" (byte_size)\n"
        result += "    :\n"
        result += "    :\n"
        result += "    );\n"
        result += "    return byte_size;\n"
        result += "}"
        return result

    def fmul(self, avreg : vreg_type, bvreg : vreg_type, cvreg : vreg_type,
             dt : adt) -> str:
        return self.asmwrap(f"vfmul.vv {cvreg},{avreg},{bvreg}")

    def fmul_vf(self, avreg : vreg_type, bfreg : freg_type, cvreg : vreg_type,
                dt : adt) -> str:
        return self.asmwrap(f"vfmul.vf {cvreg},{avreg},{bfreg}")

    @property
    def has_add_greg_voff(self) -> bool:
        return False

    def add_greg_voff(self, reg : greg_type, offset : int,
                      dt : adt) -> str:
        raise NotImplementedError("RVV doesn't have an instruction to add a vector offset to a gp register")
        
    def zero_vreg(self, vreg : vreg_type, dt : adt) -> str:
        return self.asmwrap(f"vmv.v.i {vreg},0")


    @property
    def min_load_voff(self):
        return 0

    @property
    def max_load_voff(self):
        return 0

    def load_vector(self, areg : greg_type, ignored_offset : int, vreg : vreg_type, dt : adt):
        dt_suf = self.dt_suffixes[dt]
        return self.asmwrap(f"vl{dt_suf}.v {vreg}, ({areg})")

    # I'm not seeing equivalents in RVV, I think you're supposed to do things differently
    # (LMUL > 1?), vector index?
    def load_vector_voff(self, areg : greg_type, voffset : int, vreg : vreg_type, dt : adt):
        #raise NotImplementedError("RVV has no vector loads with address offset")
        # We can still load the vector - with max_load_{imm,v}off being 0, the generator will
        # just always pass an offset of 0 and add any offset to the address register after
        if voffset != 0:
            raise NotImplementedError("RVV has no vector loads with address offset")
        return self.load_vector(areg=areg, ignored_offset=voffset, vreg=vreg, dt=dt)

    def load_vector_dist1(self, areg : greg_type, ignored_offset : int, vreg : vreg_type, dt : adt):
        dt_suf = self.dt_suffixes[dt]
        return self.asmwrap(f"vls{dt_suf}.v {vreg}, ({areg}), zero")

    def load_vector_dist1_boff(self, areg : greg_type, offset : int, vreg : vreg_type, dt : adt):
        #raise NotImplementedError("RVV has no vector loads with address offset")
        return self.load_vector_dist1(areg=areg, ignored_offset=offset, vreg=vreg, dt=dt)
    
    def load_vector_dist1_inc(self, areg : greg_type, ignored_offset : int, vreg : vreg_type, dt : adt):
        raise NotImplementedError("RVV has no vector loads with address increment")

    def store_vector(self, areg : greg_type, voffset : int, vreg : vreg_type, dt : adt):
        dt_suf = self.dt_suffixes[dt]
        return self.asmwrap(f"vs{dt_suf}.v {vreg}, ({areg})")

    def store_vector_voff(self, areg : greg_type, ignored_offset : int, vreg : vreg_type, dt : adt):
        if ignored_offset != 0:
            raise NotImplementedError("RVV has no vector stores with address offset")
        return self.store_vector(areg=areg, ignored_offset=ignored_offset, vreg=vreg, dt=dt)

    def vsetvlmax(self, reg : greg_type, dt : adt):
        dt_size = 'e'+str(adt_size(dt)*8)
        return self.asmwrap(f"vsetvli {reg}, zero, {dt_size}, m1, ta, ma")

    def load_vector_immstride(self, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, dt : adt):
        raise NotImplementedError("RVV has no load with immediate stride")

    def load_vector_gregstride(self, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, dt : adt):
        dt_suf = self.dt_suffixes[dt]
        return self.asmwrap(f"vls{dt_suf}.v {vreg}, ({areg}), {sreg}")

    def load_vector_gather(self, areg : greg_type, offvreg : vreg_type,
                           vreg : vreg_type, dt : adt,
                           indextype : ait):
        i_suf = self.it_suffixes[indextype]
        return self.asmwrap(f"vlux{i_suf}.v {vreg}, ({areg}), {offvreg}")

    def store_vector_immstride(self, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, dt : adt):
        raise NotImplementedError("RVV has no store with immediate stride")

    def store_vector_gregstride(self, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, dt : adt):
        dt_suf = self.dt_suffixes[dt]
        return self.asmwrap(f"vss{dt_suf}.v {vreg}, ({areg}), {sreg}")

    def store_vector_scatter(self, areg : greg_type, offvreg : vreg_type,
                             vreg : vreg_type, dt : adt,
                             indextype : ait):
        i_suf = self.it_suffixes[indextype]
        return self.asmwrap(f"vsux{i_suf}.v {vreg}, ({areg}), {offvreg}")

    # Unsupported functionality:
    def max_tregs(self, dt : adt):
        return 0

    def treg(self, reg_idx : int):
        raise NotImplementedError("RVV has no tiles (for now)")

    def zero_treg(self, treg : treg_type, datatype : adt):
        raise NotImplementedError("RVV has no tiles (for now)")

    def store_tile(self, areg : greg_type,
                   ignored_offset : int,
                   treg : treg_type,
                   datatype : adt):
        raise NotImplementedError("RVV has no tiles (for now)")
