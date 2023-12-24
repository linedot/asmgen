arch_flags = {
        'g++' : {
            'fma128': ['-mavx', '-mfma'],
            'fma256': ['-mavx', '-mfma'],
            'avx512': ['-mavx512f'],
            'neon': ['-march=armv8-a'],
            'sve': ['-march=armv8-a+sve'],
            'rvv': ['-march=rv64imafdcv'],
            'rvv071': ['-fail_on_purpose'],
            },
        'clang++' : {
            'fma128': ['-mavx', '-mfma'],
            'fma256': ['-mavx', '-mfma'],
            'avx512': ['-mavx512f'],
            'neon': ['-march=armv8-a'],
            'sve': ['-march=armv8-a+sve'],
            'rvv': ['-march=rv64imafdcv'],
            'rvv071': ['-mepi'],
            },
        'armclang++' : {
            'fma128': ['-fail_on_purpose'],
            'fma256': ['-fail_on_purpose'],
            'avx512': ['-fail_on_purpose'],
            'neon': ['-march=armv8-a'],
            'sve': ['-march=armv8-a+sve'],
            'rvv': ['-fail_on_purpose'],
            'rvv071': ['-fail_on_purpose'],
            },
        }

cross_archs = {
        "rvv" : "riscv64",
        "rvv071" : "riscv64",
        "neon" : "aarch64",
        "sve" : "aarch64",
        "fma128" : "x86_64",
        "fma256" : "x86_64",
        "avx512" : "x86_64",
        }

# These are probably very OS-dependent
cross_paths = {
        'x86_64' : {
            'sysroot' : '/usr/x86_64-linux-gnu',
            'interpreter' : '/usr/x86_64-linux-gnu/lib/ld-linux-x86_64.so.1',
            },
        'riscv64' : {
            'sysroot' : '/usr/riscv64-linux-gnu',
            'interpreter' : '/usr/riscv64-linux-gnu/lib/ld-linux-riscv64-lp64d.so.1',
            },
        'aarch64' : {
            'sysroot' : '/usr/aarch64-linux-gnu',
            'interpreter' : '/usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1',
            },
        }

cross_cxx_flags = {
        'x86_64' : {
            'clang++' : [f'--sysroot={cross_paths["x86_64"]["sysroot"]}', 
                         '-B',f'{cross_paths["x86_64"]["sysroot"]}/bin',
                         '-target',f'x86_64-pc-linux-gnu']
            },
        'riscv64' : {
            'clang++' : [f'--sysroot={cross_paths["riscv64"]["sysroot"]}', 
                         '-B',f'{cross_paths["riscv64"]["sysroot"]}/bin',
                         '-target',f'riscv64-pc-linux-gnu']
            },
        'aarch64' : {
            'clang++' : [f'--sysroot={cross_paths["aarch64"]["sysroot"]}', 
                         '-B',f'{cross_paths["aarch64"]["sysroot"]}/bin',
                         '-target',f'aarch64-pc-linux-gnu']
            }
        }

cross_lib_flags = {
        'x86_64' : {
            'clang++' : [f'-Wl,--dynamic-linker',f'{cross_paths["x86_64"]["interpreter"]}']
            },
        'riscv64' : {
            'clang++' : [f'-Wl,--dynamic-linker',f'{cross_paths["riscv64"]["interpreter"]}']
            },
        'aarch64' : {
            'clang++' : [f'-Wl,--dynamic-linker',f'{cross_paths["aarch64"]["interpreter"]}']
            },
        }

default_flags = {
        'g++' : ['-fdiagnostics-color=always', '-g', '-fPIC'],
        'clang++' : ['-fcolor-diagnostics', '-g', '-fPIC'],
        'armclang++' : ['-fcolor-diagnostics', '-g', '-fPIC'],
        }

stdin_flags = {
        'g++' : ['-x','c++','-'],
        'clang++' : ['-x','c++','-'],
        'armclang++' : ['-x','c++','-'],
        }

obj_flags = {
        'g++' : ['-c'],
        'clang++' : ['-c'],
        'armclang++' : ['-c'],
        }

lib_flags = {
        'g++' : ['-shared'],
        'clang++' : ['-shared'],
        'armclang++' : ['-shared'],
        }

output_flag = {
        'g++' : '-o',
        'clang++' : '-o',
        'armclang++' : '-o',
        }
