# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Various parameters and values for compilation for different ISAs
"""
arch_flags  : dict[str,dict[str,list[str]]] = {
        'g++' : {
            'fma128': ['-mavx', '-mfma'],
            'fma256': ['-mavx', '-mfma'],
            'avx512': ['-mavx512f'],
            'neon': ['-march=armv8-a'],
            'sve': ['-march=armv8-a+sve'],
            'sme': ['-march=armv8-a+sme'],
            'rvv': ['-march=rv64imafdcv'],
            'rvv071': ['-fail_on_purpose'],
            },
        'clang++' : {
            'fma128': ['-mavx', '-mfma'],
            'fma256': ['-mavx', '-mfma'],
            'avx512': ['-mavx512f'],
            'neon': ['-march=armv8-a'],
            'sve': ['-march=armv8-a+sve'],
            'sme': ['-march=armv8-a+sme'],
            'rvv': ['-march=rv64imafdcv'],
            'rvv071': ['-mepi'],
            },
        'armclang++' : {
            'fma128': ['-fail_on_purpose'],
            'fma256': ['-fail_on_purpose'],
            'avx512': ['-fail_on_purpose'],
            'neon': ['-march=armv8-a'],
            'sve': ['-march=armv8-a+sve'],
            'sme': ['-march=armv8-a+sme'],
            'rvv': ['-fail_on_purpose'],
            'rvv071': ['-fail_on_purpose'],
            },
        }

cross_archs : dict[str,str] = {
        "rvv" : "riscv64",
        "rvv071" : "riscv64",
        "neon" : "aarch64",
        "sve" : "aarch64",
        "sme" : "aarch64",
        "fma128" : "x86_64",
        "fma256" : "x86_64",
        "avx512" : "x86_64",
        }

# These are probably very OS-dependent
cross_paths : dict[str,dict[str,str]] = {
        'x86_64' : {
            'sysroot' : '/usr/x86_64-linux-gnu',
            'interpreter' : '/usr/x86_64-linux-gnu/',
            },
        'riscv64' : {
            'sysroot' : '/usr/riscv64-linux-gnu',
            'interpreter' : '/usr/riscv64-linux-gnu/',
            },
        'aarch64' : {
            'sysroot' : '/usr/aarch64-linux-gnu',
            'interpreter' : '/usr/aarch64-linux-gnu/',
            },
        }

cross_cxx_flags : dict[str,dict[str,list[str]]] = {
        'x86_64' : {
            'clang++' : [f'--sysroot={cross_paths["x86_64"]["sysroot"]}', 
                         '-B',f'{cross_paths["x86_64"]["sysroot"]}/bin',
                         '-target','x86_64-pc-linux-gnu']
            },
        'riscv64' : {
            'clang++' : [f'--sysroot={cross_paths["riscv64"]["sysroot"]}', 
                         '-B',f'{cross_paths["riscv64"]["sysroot"]}/bin',
                         '-target','riscv64-linux-gnu']
            },
        'aarch64' : {
            'clang++' : [f'--sysroot={cross_paths["aarch64"]["sysroot"]}', 
                         '-B',f'{cross_paths["aarch64"]["sysroot"]}/bin',
                         '-target','aarch64-linux-gnu']
            }
        }

cross_lib_flags : dict[str,dict[str,list[str]]] = {
        'x86_64' : {
            'clang++' : []
            },
        'riscv64' : {
            'clang++' : []
            },
        'aarch64' : {
            'clang++' : []
            },
        }

default_flags : dict[str,list[str]] = {
        'g++' : ['-fdiagnostics-color=always', '-g', '-fPIC'],
        'clang++' : ['-fcolor-diagnostics', '-g', '-fPIC'],
        'armclang++' : ['-fcolor-diagnostics', '-g', '-fPIC'],
        }

stdin_flags : dict[str,list[str]] = {
        'g++' : ['-x','c++','-'],
        'clang++' : ['-x','c++','-'],
        'armclang++' : ['-x','c++','-'],
        }

obj_flags : dict[str,list[str]] = {
        'g++' : ['-c'],
        'clang++' : ['-c'],
        'armclang++' : ['-c'],
        }

lib_flags : dict[str,list[str]] = {
        'g++' : ['-shared'],
        'clang++' : ['-shared'],
        'armclang++' : ['-shared'],
        }

output_flag : dict[str,str] = {
        'g++' : '-o',
        'clang++' : '-o',
        'armclang++' : '-o',
        }
