# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from ..operations import (
    opdna1,
    opdna1_action,
    opdna1_modifier as mod
)
from ...registers import asm_data_type as adt, adt_size, data_reg
from ..types.avx_types import (
    x86_greg,
    avx_vreg, xmm_vreg, ymm_vreg, zmm_vreg,
    avx_freg,
    avx512_mreg
)
from ..x86_opdna1.x86_opdna1_base import x86_opdna1


class avx_opdna1(opdna1):
    def __init__(self, action: opdna1_action, simd_bytes: int,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        self.action = action
        self.simd_bytes = simd_bytes
        self.asmwrap = asmwrap
        self.rpref = rpref
        self.scalar_opdna1 = x86_opdna1(action=action, asmwrap=asmwrap, rpref=rpref)

    def supported_dts(self) -> list[adt]:
        sup_dts = [
            adt.FP64,
            adt.FP32,
            adt.FP16,
            adt.BF16,
            adt.SINT64,
            adt.SINT32,
            adt.SINT16,
            adt.SINT8,
            adt.UINT64,
            adt.UINT32,
            adt.UINT16,
            adt.UINT8
        ]

        return [{'adreg': dt} for dt in sup_dts]

    def check_modifiers(self, modifiers: set[mod]):
        if mod.TINDEX in modifiers:
            raise ValueError("AVX has no ld/st with 2D tile offset indices")
        if mod.GLANE in modifiers:
            raise ValueError("AVX has no GP-reg lane ld/st")
        if mod.POSTINC in modifiers:
            raise ValueError("AVX has no postinc ld/st form")
        if mod.TOFFSET in modifiers:
            raise ValueError("AVX has no ld/st with 2D tile offsets")
        if mod.ISTRIDE in modifiers:
            raise ValueError("AVX has no ld/st with immediate strides")
        if mod.GSTRIDE in modifiers:
            raise ValueError("AVX has no ld/st with GP-reg strides")
        if mod.STRUCT in modifiers:
            raise ValueError("AVX has no structured ld/st")
        if mod.ROW in modifiers:
            raise ValueError("AVX has no row selection ld/st")
        if mod.COL in modifiers:
            raise ValueError("AVX has no column selection ld/st")
        if mod.NT in modifiers:
            raise NotImplementedError("Non-temporals for AVX not yet implemented")
        if mod.MASK in modifiers:
            raise NotImplementedError("Masked ld/st for AVX not yet implemented")

        if mod.BCAST in modifiers and self.action != opdna1_action.LOAD:
            raise ValueError("BCAST modifier can only be used with loads")

        if mod.VINDEX in modifiers:
            # TODO: look into x86/avx addressing, this might be incorrect
            if mod.VOFFSET in modifiers or mod.IOFFSET in modifiers:
                raise ValueError("VINDEX cannot be combined with IOFFSET/VOFFSET")

    def get_required_params(self, modifiers: set[mod]) -> list[set[str]]:

        required_extra_params = []
        if mod.IOFFSET in modifiers:
            required_extra_params.append({"ioffset"})

        if mod.VOFFSET in modifiers:
            required_extra_params.append({"voffset"})

        if mod.VINDEX in modifiers:
            required_extra_params.append({"vidxreg"})
            required_extra_params.append({"it"})

        if mod.ILANE in modifiers:
            required_extra_params.append({"lane"})


        return required_extra_params

    def get_operand_restrictions(self, oprnd : str) -> set[operand_restriction]:
        # No restriction on any operands
        return {}

    def get_operand_restriction_value(self, op : str,
                                      modifiers : set[mod],
                                      rstr : operand_restriction) \
      -> int|set[int]|tuple[str,int]:
        raise ValueError("No restriction {rstr} on operand {op} for AVX opd3")

        
    def get_addressing(self, areg: x86_greg, modifiers: set[mod], **kwargs) -> str:
        offset = 0
        # TODO: lost track again whether i had this be bytes or elements, double check
        if mod.IOFFSET in modifiers:
            offset = kwargs["ioffset"]
        elif mod.VOFFSET in modifiers:
            offset = kwargs["voffset"] * self.simd_bytes

        pareg = self.rpref(areg)

        return f"{offset}({pareg})" if offset != 0 else f"({pareg})"

    def get_vector_mnemonic(self, dt: adt) -> str:
        if dt == adt.FP32: return "vmovups"
        if dt == adt.FP64: return "vmovupd"
        return "vmovdqu"


    def build_bcast(self, dreg: avx_vreg, areg: x86_greg, dt: adt,
                    addressing: str) -> str:
        raise NotImplementedError("Missing broadcast implementation")

    def build_gather(self, dreg: avx_vreg, areg: x86_greg, dt:adt, **kwargs) -> str:
        raise NotImplementedError("Missing gather implementation")

    def build_scatter(self, dreg: avx_vreg, areg: x86_greg, dt:adt, **kwargs) -> str:
        raise NotImplementedError("Missing scatter implementation")

    def build_lane_load(self, dreg: avx_vreg, areg: x86_greg, dt: adt,
                        lane: int, addressing: str) -> str:
        raise NotImplementedError("Missing lane load implementation")

    def build_lane_store(self, dreg: avx_vreg, areg: x86_greg, dt: adt,
                        lane: int, addressing: str) -> str:
        raise NotImplementedError("Missing lane store implementation")


    def implementation(self, *, dregs: list[data_reg], agreg: x86_greg, a_dt: adt,
                       modifiers: set[mod], **kwargs) -> str:
        if not dregs:
            raise ValueError("No dregs provided")

        if isinstance(dregs[0], (x86_greg, avx_freg)):
            return self.scalar_opdna1(dregs=dregs, areg=agreg, dt=a_dt,
                                      modifiers=modifiers, **kwargs)

        if not all(isinstance(r, avx_vreg) for r in dregs):
            raise ValueError("AVX opdna1: All data registers must be vector registers")


        dreg = dregs[0]

        addressing = self.get_addressing(agreg, modifiers, **kwargs)


        if mod.BCAST in modifiers:
            return self.asmwrap(self.build_bcast(dreg, agreg, a_dt, addressing))

        if mod.VINDEX in modifiers:
            if self.action == opdna1_action.LOAD:
                return self.asmwrap(self.build_gather(dreg, agreg, a_dt, **kwargs))
            if self.action == opdna1_action.STORE:
                return self.asmwrap(self.build_scatter(dreg, agreg, a_dt, **kwargs))

            # Potentially adding prefetches or something else in the future
            raise ValueError(f"Action {self.action} with VINDEX not implemented")

        if mod.ILANE in modifiers:
            lane = kwargs["lane"]
            if self.action == opdna1_action.LOAD:
                return self.asmwrap(self.build_lane_load(dreg, agreg, a_dt, lane, addressing))
            else:
                return self.asmwrap(self.build_lane_store(dreg, agreg, a_dt, lane, addressing))

        pdreg = self.rpref(dreg)
        inst = self.get_vector_mnemonic(a_dt)
        if self.action == opdna1_action.LOAD:
            return self.asmwrap(f"{inst} {addressing}, {pdreg}")
        else:
            return self.asmwrap(f"{inst} {pdreg}, {addressing}")


class avx128_opdna1(avx_opdna1):
    def __init__(self, action: opdna1_action,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action, simd_bytes=16, asmwrap=asmwrap, rpref=rpref)

    def build_lane_load(self, dreg: xmm_vreg, areg: x86_greg, dt: adt,
                        lane: int, addressing: str) -> str:
        size = adt_size(dt)
        pdreg = self.rpref(dreg)

        if dt == adt.FP64:
            if lane == 0: return f"vmovsd {addressing}, {pdreg}"
            if lane == 1: return f"vmovhpd {addressing}, {pdreg}, {pdreg}"
            raise ValueError(f"FP64 lane {lane} out of bounds")
        # Float 32 uses vinsertps (syntax is slightly quirky:
        # imm8 dictates dst/src/zero mask)
        # Imm8 format for memory load: (lane << 4)
        if dt == adt.FP32:
            if lane == 0: return f"vmovss {addressing}, {pdreg}"
            imm8 = lane << 4
            return f"vinsertps ${imm8}, {addressing}, {pdreg}, {pdreg}"
            
        # Everything else uses integer domain inserts (vpinsrb/w/d/q)
        if size == 1: return f"vpinsrb ${lane}, {addressing}, {pdreg}, {pdreg}"
        if size == 2: return f"vpinsrw ${lane}, {addressing}, {pdreg}, {pdreg}"
        if size == 4: return f"vpinsrd ${lane}, {addressing}, {pdreg}, {pdreg}"
        if size == 8: return f"vpinsrq ${lane}, {addressing}, {pdreg}, {pdreg}"
        
        raise ValueError(f"Unsupported lane load size: {size}")

    def build_lane_store(self, dreg: xmm_vreg, areg: x86_greg, dt: adt,
                         lane: int, addressing: str) -> str:
        size = adt_size(dt)
        pdreg = self.rpref(dreg)

        if dt == adt.FP64:
            if lane == 0: return f"vmovsd {pdreg},{addressing}"
            if lane == 1: return f"vmovhpd {pdreg},{addressing}"
            raise ValueError(f"FP64 lane {lane} out of bounds")
        
        # Float 32 uses extractps
        if dt == adt.FP32:
            imm8 = lane << 4
            return f"vextractps ${imm8}, {pdreg}, {addressing}"
            
        # Integer domain extracts (vpextrb/w/d/q)
        if size == 1: return f"vpextrb ${lane}, {pdreg}, {addressing}"
        if size == 2: return f"vpextrw ${lane}, {pdreg}, {addressing}"
        if size == 4: return f"vpextrd ${lane}, {pdreg}, {addressing}"
        if size == 8: return f"vpextrq ${lane}, {pdreg}, {addressing}"
        
        raise ValueError(f"Unsupported lane store size: {size}")

    def build_bcast(self, dreg: xmm_vreg, areg: x86_greg, dt: adt, addressing: str) -> str:
        suf = "ss" if adt_size(dt) == 4 else "sd"
        pdreg = self.rpref(dreg)
        return f"vbroadcast{suf} {addressing}, {pdreg}" 
        
    def build_gather(self, dreg: xmm_vreg, areg: x86_greg, dt: adt, **kwargs) -> str:
        suf = "ps" if adt_size(dt) == 4 else "pd"
        isuf = "d" if adt_size(kwargs["it"]) == 4 else "q" 
        vidxreg = kwargs["vidxreg"]

        pareg = self.rpref(areg)
        pvidxreg = self.rpref(vidxreg)
        pdreg = self.rpref(dreg)
        
        return f"vgather{isuf}{suf} ({pareg}), {pvidxreg}, {pdreg}"

    def build_scatter(self, dreg: xmm_vreg, areg: x86_greg, dt: adt, **kwargs) -> str:
        raise NotImplementedError("AVX/AVX2 has no store with vector register stride (Scatter)")

class avx256_opdna1(avx128_opdna1):
    def __init__(self, action: opdna1_action,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        # AVX256 is functionally identical to 128 here, just with 32-byte offsets
        super().__init__(action=action, asmwrap=asmwrap, rpref=rpref)
        self.simd_bytes = 32

class avx512_opdna1(avx_opdna1):
    def __init__(self, action: opdna1_action,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action, simd_bytes=64, asmwrap=asmwrap, rpref=rpref)

    def build_bcast(self, dreg: zmm_vreg, areg: x86_greg, dt: adt, addressing: str) -> str:
        suf = "ss" if adt_size(dt) == 4 else "sd"

        pdreg = self.rpref(dreg)
        return f"vbroadcast{suf} {addressing}, {pdreg}"

    def build_gather(self, dreg: zmm_vreg, areg: x86_greg, dt: adt, **kwargs) -> str:
        suf = "ps" if adt_size(dt) == 4 else "pd"
        isuf = "d" if adt_size(kwargs["it"]) == 4 else "q" 
        vidxreg = kwargs["vidxreg"]

        pareg = self.rpref(areg)
        pvidxreg = self.rpref(vidxreg)
        
        addressing = f"({pareg},{pvidxreg},1)"
        maskreg = self.rpref(avx512_mreg(2))
        masksuf = "w" if adt_size(dt) == 4 else "q"

        pdreg = self.rpref(dreg)
        
        return (f"kxnor{masksuf} {maskreg}, {maskreg}, {maskreg}\n"
                f"vgather{isuf}{suf} {addressing}, {pdreg}{{{maskreg}}}")

    def build_scatter(self, dreg: zmm_vreg, areg: x86_greg, dt: adt, **kwargs) -> str:
        suf = "ps" if adt_size(dt) == 4 else "pd"
        isuf = "d" if adt_size(kwargs["it"]) == 4 else "q" 
        vidxreg = kwargs["vidxreg"]

        pareg = self.rpref(areg)
        pvidxreg = self.rpref(vidxreg)
        
        addressing = f"({pareg},{pvidxreg},1)"
        maskreg = self.rpref(avx512_mreg(2))
        masksuf = "w" if adt_size(dt) == 4 else "q"
        
        return (f"kxnor{masksuf} {maskreg}, {maskreg}, {maskreg}\n"
                f"vscatter{isuf}{suf} {dreg}, {addressing}{{{maskreg}}}")
