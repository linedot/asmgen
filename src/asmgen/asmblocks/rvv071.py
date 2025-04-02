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

class rvv071(riscv64):

    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = rvv_vreg
    treg_type : TypeAlias = treg

    dt_suffixes = {
            adt.DOUBLE : "e",
            adt.SINGLE : "w",
            adt.HALF   : "h",
            }
    it_suffixes = {
            ait.INT64 : "ei64",
            ait.INT32 : "ei32",
            ait.INT16 : "ei16",
            ait.INT8  : "ei8",
            }

    def __init__(self):
        super(rvv071, self).__init__()
        self.fma = rvv_fma(asmwrap=self.asmwrap)
        self.fmul = rvv_fmul(asmwrap=self.asmwrap)

    def supportedby_cpuinfo(self, cpuinfo : str) -> bool:
         isa_idx = cpuinfo.find("rv64")
         if -1 == isa_idx:
             return False
         isa_idx = isa_idx+4
         extensions = cpuinfo[isa_idx:].split()[0]
         return "v" in extensions

    def isaquirks(self, rt : reg_tracker, dt : adt):
        tmpreg_idx = rt.reserve_any_reg("greg")
        tmpreg = self.greg(tmpreg_idx)
        asmblock = self.vsetvlmax(tmpreg, dt)
        rt.unuse_reg("greg", tmpreg_idx)
        return asmblock

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

    def simd_size_to_greg(self, reg : greg_type, dt : adt):
        asmblock  = self.asmwrap(f"csrr {reg}, vl")
        asmblock += self.asmwrap(f"slli {reg}, {reg}, {adt_size(dt).bit_length()-1}")
        return asmblock

    @property
    def c_simd_size_function(self):
        result  = "size_t get_simd_size() {\n"
        result += "    size_t byte_size = 0;\n"
        result += "    __asm__ volatile(\n"
        result += "        "+self.asmwrap("addi t0, zero, 60")
        result += "        "+self.asmwrap("slli t0, t0, 5")
        result += "        "+self.asmwrap("vsetvli %[byte_size], t0, e8, m1")
        result += "    : [byte_size] \"=r\" (byte_size)\n"
        result += "    :\n"
        result += "    : \"t0\"\n"
        result += "    );\n"
        result += "    return byte_size;\n"
        result += "}"
        return result


    # TODO: These don't change between 0.7.1 and 1.0, could be deduplicated

    def fmul(self, avreg : vreg_type, bvreg : vreg_type, cvreg : vreg_type,
             dt : adt) -> str:
        return self.asmwrap(f"vfmul.vv {cvreg},{avreg},{bvreg}")

    def fmul_vf(self, avreg : vreg_type, bfreg : freg_type, cvreg : vreg_type,
                dt : adt) -> str:
        return self.asmwrap(f"vfmul.vf {cvreg},{avreg},{bfreg}")

    def fma(self, avreg : vreg_type, bvreg : vreg_type, cvreg : vreg_type,
            dt : adt) -> str:
        return self.asmwrap(f"vfmacc.vv {cvreg},{avreg},{bvreg}")

    def fma_np(self, avreg : vreg_type, bvreg : vreg_type, cvreg : vreg_type,
            dt : adt) -> str:
        return self.asmwrap(f"vfnmsac.vv {cvreg},{avreg},{bvreg}")

    def fma_vf(self, avreg : vreg_type, bfreg : freg_type, cvreg : vreg_type,
               dt : adt) -> str:
        return self.asmwrap(f"vfmacc.vf {cvreg},{bfreg},{avreg}")

    def fma_np_vf(self, avreg : vreg_type, bfreg : freg_type, cvreg : vreg_type,
               dt : adt) -> str:
        return self.asmwrap(f"vfnmsac.vf {cvreg},{bfreg},{avreg}")

    def fma_idx(self, avreg : vreg_type, bvreg : vreg_type, cvreg : vreg_type,
                idx : int, dt : adt) -> str:
        raise NotImplementedError("RVV doesn't have an indexed FMA")

    def fma_np_idx(self, avreg : vreg_type, bfreg : freg_type, cvreg : vreg_type,
                   idx : int, dt : adt) -> str:
        raise NotImplementedError("RVV doesn't have an indexed FMA")

    @property
    def has_add_greg_voff(self) -> bool:
        return False

    def add_greg_voff(self, reg : greg_type, offset : int, dt : adt):
        raise NotImplementedError("RVV doesn't have an instruction to add a vector offset to a gp register")
        
    def zero_vreg(self, reg : greg_type, dt : adt):
        return self.asmwrap(f"vmv.v.i {reg},0")

    def vreg(self, reg_idx : int) -> rvv_vreg:
        return rvv_vreg(reg_idx)

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
        if ignored_offset != 0:
            raise NotImplementedError("RVV has no vector loads with address offset")
        return self.load_vector(areg=areg, ignored_offset=voffset, vreg=vreg, dt=dt)

    def load_vector_dist1(self, areg : greg_type, offset : int, vreg : vreg_type, dt : adt):
        dt_suf = self.dt_suffixes[dt]
        # This is slow on a certain architecture and doesn't support offsets
        return self.asmwrap(f"vls{dt_suf}.v {vreg}, ({areg}), zero")

    def load_vector_dist1_boff(self, areg : greg_type, offset : int, vreg : vreg_type, dt : adt):
        if offset != 0:
            raise NotImplementedError("RVV has no vector loads with address offset")
        return self.load_vector_dist1(areg=areg, offset=offset, vreg=vreg, dt=dt)
    
    def load_vector_dist1_inc(self, areg : greg_type, ignored_offset : int, vreg : vreg_type, dt : adt):
        raise NotImplementedError("RVV has no vector loads with address increment")

    def store_vector(self, areg : greg_type, voffset : int, vreg : vreg_type, dt : adt):

        dt_suf = self.dt_suffixes[dt]
        return self.asmwrap(f"vs{dt_suf}.v {vreg}, ({areg})")

    def store_vector_voff(self, areg : greg_type, voffset : int, vreg : vreg_type, dt : adt):
        if voffset != 0:
            raise NotImplementedError("RVV has no vector stores with address offset")
        self.store_vector(areg=areg, voffset=voffset, vreg=vreg, dt=dt)

    def vsetvlmax(self, reg : greg_type, dt : adt):
        dt_size = 'e'+str(adt_size(dt)*8)
        asmblock  = "        "+self.asmwrap(f"addi {reg}, zero, 60")
        asmblock += "        "+self.asmwrap(f"slli {reg}, {reg}, {6-adt_size(dt).bit_length()}")
        asmblock += self.asmwrap(f"vsetvli {reg}, {reg}, {dt_size}, m1")
        return asmblock

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
        dt_suf = self.it_suffixes[indextype]
        return self.asmwrap(f"vlx{dt_suf}.v {vreg}, ({areg}), {offvreg}")

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
        dt_suf = self.it_suffixes[indextype]
        return self.asmwrap(f"vsux{dt_suf}.v {vreg}, ({areg}), {offvreg}")

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
