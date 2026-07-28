# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Example generating an AXPY function
"""

import sys
import importlib

from asmgen.registers import (
        adt_size,
        asm_data_type as adt,
        reg_tracker
)
from asmgen.asmblocks.noarch import asmgen
from asmgen.callconv.fngen import fngen
from asmgen.asmblocks.noarch import comparison
from asmgen.asmblocks.op import opd3_modifier as opd3_mod
from asmgen.asmblocks.op.operand import register_type as op_rt

isa_modules = {
        "fma128" : "avx_fma",
        "fma256" : "avx_fma",
        "avx512" : "avx_fma",
        }

def get_simplest_signature(signatures, dt, req_rt=None, require_mod=None):
    """
    Finds the simplest matching signature for a given datatype.
    Optionally enforces register type and/or a specific modifier.
    """
    valid_sigs = []
    for s in signatures:
        if s.operands['adreg'].dt == dt:
            # Check register type (e.g. ensure we don't accidentally pick a scalar signature)
            if req_rt and s.operands['adreg'].rtype != req_rt:
                continue

            if require_mod and require_mod not in s.modifiers:
                continue

            valid_sigs.append(s)

    if not valid_sigs:
        return None

    # Sort by number of modifiers; fewer modifiers = more "basic" instruction
    return sorted(valid_sigs, key=lambda s: len(s.modifiers))[0]

def build_kwargs_from_modifiers(modifiers, mreg):
    """
    Dynamically generates the required kwargs dict based on active modifiers.
    """
    kw = {}
    for m in modifiers:
        if m.name == "MASK":
            kw['amreg'] = mreg
            kw['bmreg'] = mreg # Passed safely just in case it's an opd3 that requires both
        elif m.name == "IOFFSET":
            kw['ioffset'] = 0
    return kw

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
        ("mreg",gen.max_mregs)
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
    n_idx          = rt.aliased_regs["greg"]["n"]

    asmheader = (
             ".section .text\n"
            f".global {fnname}\n"
            f"{fnname}:\n  "
        )

    innerblock = ""

    addr_x = gen.greg(addr_x_idx)
    addr_y = gen.greg(addr_y_idx)
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

    # -------------------------------------------------------------------------
    # 1. Declaratively resolve operation signatures
    # -------------------------------------------------------------------------

    # Look for an FMA signature that supports scalar-vector (VF)
    fma_sig = get_simplest_signature(gen.fma.get_signatures(), dt,
                                     req_rt=op_rt.VEC, require_mod=opd3_mod.VF)

    if fma_sig:
        can_vf = True
    else:
        # Fallback to standard vector-vector FMA
        can_vf = False
        fma_sig = get_simplest_signature(gen.fma.get_signatures(), dt,
                                         req_rt=op_rt.VEC)

    # Get standard load and store signatures
    ld_sig = get_simplest_signature(gen.load.get_signatures(), dt,
                                    req_rt=op_rt.VEC)
    st_sig = get_simplest_signature(gen.store.get_signatures(), dt,
                                    req_rt=op_rt.VEC)

    fma_mods = fma_sig.modifiers
    ld_mods = ld_sig.modifiers
    st_mods = st_sig.modifiers

    # -------------------------------------------------------------------------
    # 2. Check mask requirements and allocate if necessary
    # -------------------------------------------------------------------------

    needs_mask = any(m.name == "MASK" for m in fma_mods | ld_mods | st_mods)
    mreg = None

    if needs_mask:
        m_idx = rt.reserve_any_reg("mreg")
        mreg = gen.mreg(m_idx)
        # Initialize to all-true for basic unpredicated loop behavior
        if hasattr(gen, "ptrue"):
            innerblock += gen.ptrue(reg=mreg, dt=dt) # SVE
        elif hasattr(gen, "init_mask_all"):
            innerblock += gen.init_mask_all(mreg=mreg, dt=dt)

    # Pre-build the kwargs dictated by the chosen signatures
    ld_kwargs = build_kwargs_from_modifiers(ld_mods, mreg)
    st_kwargs = build_kwargs_from_modifiers(st_mods, mreg)
    fma_kwargs = build_kwargs_from_modifiers(fma_mods, mreg)

    # -------------------------------------------------------------------------
    # 3. Setup scalar/vector broadcast based on capability
    # -------------------------------------------------------------------------

    if not can_vf:
        alpha_vreg_idx = rt.reserve_any_reg("vreg")
        alpha_vreg = gen.vreg(alpha_vreg_idx)
        innerblock += gen.fill_vector(sreg=alpha, vreg=alpha_vreg, dt=dt)
        breg = alpha_vreg
    else:
        breg = alpha

    innerblock += gen.label(label="loop")

    # -------------------------------------------------------------------------
    # 4. Generate Core Loop using declarative opdna1 / opd3 APIs
    # -------------------------------------------------------------------------

    innerblock += gen.load(dregs=[x], areg=addr_x, dt=dt, modifiers=ld_mods, **ld_kwargs)
    innerblock += gen.load(dregs=[y], areg=addr_y, dt=dt, modifiers=ld_mods, **ld_kwargs)

    innerblock += gen.fma(
            adreg=x,
            bdreg=breg,
            cdreg=y,
            a_dt=dt, b_dt=dt, c_dt=dt,
            modifiers=fma_mods,
            **fma_kwargs)

    innerblock += gen.store(dregs=[y], areg=addr_y, dt=dt, modifiers=st_mods, **st_kwargs)

    # Loop pointer updates
    if "vlen" in rt.aliased_regs["greg"]:
        innerblock += gen.add_greg_greg(dst=addr_x, reg1=addr_x, reg2=vlen)
        innerblock += gen.add_greg_greg(dst=addr_y, reg1=addr_y, reg2=vlen)
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
