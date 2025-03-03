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
