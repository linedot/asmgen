# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX operations with n data operand and 1 address operand (load/store)
"""
from typing import Callable, Any

from ..op import (
    opdna1,
    opdna1_modifier as mod,
    opdna1_action,
    operation_signature
)
from ...registers import (
    asm_data_type as adt,
    adt_size,
    ait_size,
    data_reg
)
from ..types.avx_types import (
    x86_greg,
    avx_vreg, xmm_vreg, ymm_vreg, zmm_vreg,
    avx_freg,
    avx512_mreg
)
from ..x86_opdna1.x86_opdna1_base import x86_opdna1

from .signatures import make_avx_opdna1_signatures

class avx_opdna1(opdna1):
    """
    AVX instruction with n data operands and 1 address operand.
    Uses x86_opdna1 to automatically handle scalar routing.
    """
    def __init__(self, action: opdna1_action, simd_bytes: int,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        self.action = action
        self.simd_bytes = simd_bytes
        self.asmwrap = asmwrap
        self.rpref = rpref
        self.scalar_opdna1 = x86_opdna1(action=action, asmwrap=asmwrap, rpref=rpref)

        self.signatures = []

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    def diagnose_failure(self, modifiers : set[mod],
                         kwargs : dict[str,Any],
                         dts : dict[str, adt]) -> list[operation_signature]:

        unsupported_mods = {
            mod.TINDEX:  (ValueError, "Base X86 has no ld/st with 2D tile offset indices"),
            mod.GLANE:   (ValueError, "Base X86 has no GP-reg lane ld/st"),
            mod.POSTINC: (ValueError, "Base X86 has no postinc ld/st"),
            mod.TOFFSET: (ValueError, "Base X86 has no ld/st with 2D tile offsets"),
            mod.VOFFSET: (ValueError, "Base X86 has no ld/st with vector offsets"),
            mod.ISTRIDE: (ValueError, "Base X86 has no ld/st with immediate strides"),
            mod.GSTRIDE: (ValueError, "Base X86 has no ld/st with GP-reg strides"),
            mod.STRUCT:  (ValueError, "Base X86 has no structured ld/st"),
            mod.ROW:     (ValueError, "Base X86 has no row selection ld/st"),
            mod.COL:     (ValueError, "Base X86 has no column selection ld/st"),
            mod.NT:      (NotImplementedError, "Non-temporals for Base X86 not yet implemented"),
        }
        for m, (exc_type, msg) in unsupported_mods.items():
            if m in modifiers:
                raise exc_type(msg)

        required_params = {
            mod.IOFFSET : ['ioffset'],
            mod.VOFFSET : ['voffset'],
            mod.VINDEX  : ['vidxreg','it'],
            mod.ILANE   : ['lane'],
            mod.GOFFSET : ['offreg'],
        }
        for m, plist in required_params.items():
            for p in plist:
                if m in modifiers and p not in kwargs:
                    raise ValueError(f"{m.name} modifier requires '{p}' parameter")


    def get_addressing(self, areg: x86_greg, modifiers: set[mod], **kwargs) -> str:
        """
        Constructs the addressing string

        :param areg: address register
        :param modifiers: operation modifiers

        :return: string containing the addressing
        """
        offset = 0
        if mod.IOFFSET in modifiers:
            offset = kwargs["ioffset"]
        elif mod.VOFFSET in modifiers:
            offset = kwargs["voffset"] * self.simd_bytes

        pareg = self.rpref(areg)

        return f"{offset}({pareg})" if offset != 0 else f"({pareg})"

    def get_vector_mnemonic(self, dt: adt) -> str:
        """
        Get type dependent mnemonic

        :param dt: data type
        """
        if dt == adt.FP32:
            return "vmovups"
        if dt == adt.FP64:
            return "vmovupd"
        return "vmovdqu"


    def build_bcast(self, dreg: avx_vreg, areg: x86_greg, dt: adt,
                    addressing: str) -> str:
        """
        Generate broadcast instruction

        :param dreg: vector register to broadcast into
        :param areg: gp register with the base address
        :param dt: element data type
        :param addressing: string containing the addressing
        """
        raise NotImplementedError("Missing broadcast implementation")

    def build_gather(self, dreg: avx_vreg, mreg: avx_vreg|avx512_mreg,
                     areg: x86_greg, dt:adt, **kwargs) -> str:
        """
        Generate gather instruction

        :param dreg: vector register to gather into
        :param areg: gp register with the base address
        :param mreg: mask register (xmmN/ymmN for AVX2, kN for AVX512)
        :param dt: element data type
        """
        raise NotImplementedError("Missing gather implementation")

    def build_scatter(self, dreg: avx_vreg, mreg: avx_vreg|avx512_mreg,
                      areg: x86_greg, dt:adt, **kwargs) -> str:
        """
        Generate scatter instruction

        :param dreg: vector register to scatter from
        :param areg: gp register with the base address
        :param mreg: mask register (xmmN/ymmN for AVX2, kN for AVX512)
        :param dt: element data type
        """
        raise NotImplementedError("Missing scatter implementation")

    def build_lane_load(self, dreg: avx_vreg, areg: x86_greg, dt: adt,
                        lane: int, addressing: str) -> str:
        """
        Generate scalar load into a vector lane

        :param dreg: vector register to load into
        :param areg: gp register with the base address
        :param dt: element data type
        :param lane: vector lane/element index
        """
        raise NotImplementedError("Missing lane load implementation")

    def build_lane_store(self, dreg: avx_vreg, areg: x86_greg, dt: adt,
                        lane: int, addressing: str) -> str:
        """
        Generate scalar store from a vector lane

        :param dreg: vector register to store from
        :param areg: gp register with the base address
        :param dt: element data type
        :param lane: vector lane/element index
        """
        raise NotImplementedError("Missing lane store implementation")

    def build_vindex(self, dreg : avx_vreg, mreg : avx_vreg|avx512_mreg,
                     areg : x86_greg, dt : adt, **kwargs) -> str:
        """
        Build gather/scatter

        :param dreg: vector register to use
        :param mreg: mask register to use
        :param areg: gp register with the base address
        :param dt: element data type
        """
        if self.action == opdna1_action.LOAD:
            return self.asmwrap(
                    self.build_gather(dreg, mreg, areg, dt, **kwargs))
        if self.action == opdna1_action.STORE:
            return self.asmwrap(
                    self.build_scatter(dreg, mreg, areg, dt, **kwargs))

        # Potentially adding prefetches or something else in the future
        raise ValueError(f"Action {self.action} with VINDEX not valid")


    # It's fine
    # pylint: disable-next=too-many-return-statements
    def implementation(self, *, dregs: list[data_reg], agreg: x86_greg, a_dt: adt,
                       modifiers: set[mod], **kwargs) -> str:
        if not dregs:
            raise ValueError("No dregs provided")

        if isinstance(dregs[0], (x86_greg, avx_freg)):
            return self.scalar_opdna1(dregs=dregs, areg=agreg, dt=a_dt,
                                      modifiers=modifiers, **kwargs)

        dreg = dregs[0]

        addressing = self.get_addressing(agreg, modifiers, **kwargs)


        if mod.BCAST in modifiers:
            return self.asmwrap(self.build_bcast(dreg, agreg, a_dt, addressing))

        if mod.VINDEX in modifiers:
            return self.build_vindex(dreg, kwargs['amreg'], agreg, a_dt, **kwargs)

        if mod.ILANE in modifiers:
            lane = kwargs["lane"]
            if not isinstance(dreg, xmm_vreg):
                xmm_alias = xmm_vreg(dreg.idx)
            else:
                xmm_alias = dreg

            if self.action == opdna1_action.LOAD:
                return self.asmwrap(
                        self.build_lane_load(xmm_alias, agreg, a_dt, lane, addressing))
            if self.action == opdna1_action.STORE:
                return self.asmwrap(
                        self.build_lane_store(xmm_alias, agreg, a_dt, lane, addressing))

            raise ValueError(f"Action {self.action} with ILANE not valid")

        pdreg = self.rpref(dreg)
        inst = self.get_vector_mnemonic(a_dt)
        if self.action == opdna1_action.LOAD:
            return self.asmwrap(f"{inst} {addressing}, {pdreg}")
        if self.action == opdna1_action.STORE:
            return self.asmwrap(f"{inst} {pdreg}, {addressing}")

        raise ValueError(f"Action {self.action} not valid")


class avx128_opdna1(avx_opdna1):
    """
    AVX2 128bit opdna1 implementation
    """
    ie_suffixes = {1:'b',2:'w',4:'d',8:'q'}
    def __init__(self, action: opdna1_action,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action, simd_bytes=16, asmwrap=asmwrap, rpref=rpref)

        self.signatures = make_avx_opdna1_signatures(action=self.action, is_avx512=False)
        self.signatures.extend(self.scalar_opdna1.get_signatures())

    def build_lane_load(self, dreg: xmm_vreg, areg: x86_greg, dt: adt,
                        lane: int, addressing: str) -> str:
        size = adt_size(dt)
        pdreg = self.rpref(dreg)

        if dt == adt.FP64:
            if lane == 0:
                return f"vmovsd {addressing}, {pdreg}"
            if lane == 1:
                return f"vmovhpd {addressing}, {pdreg}, {pdreg}"
            raise ValueError(f"FP64 lane {lane} out of bounds")
        # Float 32 uses vinsertps (syntax is slightly quirky:
        # imm8 dictates dst/src/zero mask)
        # Imm8 format for memory load: (lane << 4)
        if dt == adt.FP32:
            if lane == 0:
                return f"vmovss {addressing}, {pdreg}"
            imm8 = lane << 4
            return f"vinsertps ${imm8}, {addressing}, {pdreg}, {pdreg}"

        # Everything else uses integer domain inserts (vpinsrb/w/d/q)

        return f"vpinsr{self.ie_suffixes[size]} ${lane}, {addressing}, {pdreg}, {pdreg}"

    def build_lane_store(self, dreg: xmm_vreg, areg: x86_greg, dt: adt,
                         lane: int, addressing: str) -> str:
        size = adt_size(dt)
        pdreg = self.rpref(dreg)

        if dt == adt.FP64:
            if lane == 0:
                return f"vmovsd {pdreg},{addressing}"
            if lane == 1:
                return f"vmovhpd {pdreg},{addressing}"
            raise ValueError(f"FP64 lane {lane} out of bounds")

        # Float 32 uses extractps
        if dt == adt.FP32:
            imm8 = lane << 4
            return f"vextractps ${imm8}, {pdreg}, {addressing}"

        # Integer domain extracts (vpextrb/w/d/q)
        return f"vpextr{self.ie_suffixes[size]} ${lane}, {pdreg}, {addressing}"

    def build_bcast(self, dreg: xmm_vreg, areg: x86_greg, dt: adt, addressing: str) -> str:
        suf = "ss" if adt_size(dt) == 4 else "sd"
        pdreg = self.rpref(dreg)
        return f"vbroadcast{suf} {addressing}, {pdreg}"

    def build_gather(self, dreg: xmm_vreg|ymm_vreg, mreg : xmm_vreg|ymm_vreg,
                     areg: x86_greg, dt: adt, **kwargs) -> str:
        suf = "ps" if adt_size(dt) == 4 else "pd"
        isuf = "d" if ait_size(kwargs["it"]) == 4 else "q"
        vidxreg = kwargs["vidxreg"]

        pareg = self.rpref(areg)
        pvidxreg = self.rpref(vidxreg)
        pdreg = self.rpref(dreg)
        pmreg = self.rpref(mreg)

        return f"vgather{isuf}{suf} {pmreg}, ({pareg},{pvidxreg},{adt_size(dt)}), {pdreg}"

    def build_scatter(self, dreg: xmm_vreg, mreg : xmm_vreg,
                      areg: x86_greg, dt: adt, **kwargs) -> str:
        raise NotImplementedError("AVX/AVX2 has no store with vector register stride (Scatter)")

class avx256_opdna1(avx128_opdna1):
    """
    AVX2 256bit opdna1 implementation
    """
    def __init__(self, action: opdna1_action,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        # AVX256 is functionally identical to 128 here, just with 32-byte offsets
        super().__init__(action=action, asmwrap=asmwrap, rpref=rpref)
        self.simd_bytes = 32

        self.signatures = make_avx_opdna1_signatures(action=self.action, is_avx512=False)
        self.signatures.extend(self.scalar_opdna1.get_signatures())

class avx512_opdna1(avx_opdna1):
    """
    AVX512 opdna1 implementation
    """
    def __init__(self, action: opdna1_action,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action, simd_bytes=64, asmwrap=asmwrap, rpref=rpref)

        self.signatures = make_avx_opdna1_signatures(action=self.action, is_avx512=True)
        self.signatures.extend(self.scalar_opdna1.get_signatures())

    def build_bcast(self, dreg: zmm_vreg, areg: x86_greg, dt: adt, addressing: str) -> str:
        suf = "ss" if adt_size(dt) == 4 else "sd"

        pdreg = self.rpref(dreg)
        return f"vbroadcast{suf} {addressing}, {pdreg}"

    def build_gather(self, dreg: zmm_vreg, mreg : avx512_mreg,
                     areg: x86_greg, dt: adt, **kwargs) -> str:
        suf = "ps" if adt_size(dt) == 4 else "pd"
        isuf = "d" if ait_size(kwargs["it"]) == 4 else "q"
        vidxreg = kwargs["vidxreg"]

        pareg = self.rpref(areg)
        pvidxreg = self.rpref(vidxreg)

        addressing = f"({pareg},{pvidxreg},{adt_size(dt)})"
        maskreg = self.rpref(mreg)
        #masksuf = "w" if adt_size(dt) == 4 else "q"

        pdreg = self.rpref(dreg)

        #return (f"kxnor{masksuf} {maskreg}, {maskreg}, {maskreg}\n"
        #        f"vgather{isuf}{suf} {addressing}, {pdreg}{{{maskreg}}}")
        return f"vgather{isuf}{suf} {addressing}, {pdreg}{{{maskreg}}}"

    def build_scatter(self, dreg: zmm_vreg, mreg: avx512_mreg,
                      areg: x86_greg, dt: adt, **kwargs) -> str:
        suf = "ps" if adt_size(dt) == 4 else "pd"
        isuf = "d" if ait_size(kwargs["it"]) == 4 else "q"
        vidxreg = kwargs["vidxreg"]

        pareg = self.rpref(areg)
        pvidxreg = self.rpref(vidxreg)

        addressing = f"({pareg},{pvidxreg},{adt_size(dt)})"
        maskreg = self.rpref(mreg)
        #masksuf = "w" if adt_size(dt) == 4 else "q"

        pdreg = self.rpref(dreg)

        #return (f"kxnor{masksuf} {maskreg}, {maskreg}, {maskreg}\n"
        #        f"vscatter{isuf}{suf} {dreg}, {addressing}{{{maskreg}}}")
        return f"vscatter{isuf}{suf} {pdreg}, {addressing}{{{maskreg}}}"
