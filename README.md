<!-------------------------------------------------------------------------------
 SPDX-License-Identifier: MIT OR GPL-3.0-or-later
 Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
 Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
 --------------------------------------------------------------------------------->
# ASMGEN

Low-level tool for generating GAS-style inline assembly for use in C/C++ code (Or other languages that have the same syntax for inline assembly).

Supports a subset of instructions from the following ISAs:

- x86
    - AVX2+FMA 128bit
    - AVX2+FMA 256bit
    - AVX512
- Arm
    - NEON
    - SVE
    - SME
- RISCV
    - RVV 1.0
    - RVV 0.7.1


# License

asmgen is distributed under the terms of both the MIT license and the GNU General Public License v3.0. Users may choose either license, at their option.

All new contributions must be made under both the MIT and GNU General Public License v3.0.

See LICENSE-GPL-3.0, LICENSE-MIT for details.

SPDX-License-Identifier: MIT OR GPL-3.0-or-later
