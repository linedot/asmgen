from abc import abstractmethod
from itertools import permutations
from subprocess import Popen, PIPE
import logging
import os
import platform
import shutil

class rvvparams:
    def __init__(self, vspec : str, vlen : int, dlen : int):
        self.vspec = vspec
        self.vlen = vlen
        self.dlen = dlen

class sveparams:
    def __init__(self, vlen : int):
        self.vlen = vlen


class emulator:
    def __init__(self):
        self.find_paths()
        pass

    @abstractmethod
    def run(executable : str):
        raise NotImplementedError("emulator doesn't implement run")


class qemu_static_emulator(emulator):
    def __init__(self, arch : str, cpuargs : str):
        suffix_permutations = set(permutations(["static",arch])) |\
                              set(permutations(["static","user",arch]))

        possible_exe_names = ["qemu-" + "-".join(p) for p in suffix_permutations]

        self.exe = ""
        for name in possible_exe_names:
            executable = shutil.which(name)
            if None != executable:
                self.exe = executable
                break
        if not self.exe:
            raise RuntimeError(f"Could not find a static qemu that supports {arch} architecture")
        

        # Find library dir 
        possible_roots = [
                f"/usr/{arch}-linux-gnu",
                f"/usr/local/{arch}-linux-gnu",
                f"/usr/{arch}-unknown-gnu",
                f"/usr/local/{arch}-unknown-gnu",
                ]

        more_roots = os.getenv("CROSS_ROOT_PATH",default=None)
        if None != more_roots:
            more_roots = more_roots.split(":")
            possible_roots = [os.path.join(r,f"{arch}-linux-gnu") for r in more_roots]+\
                             [os.path.join(r,f"{arch}-unknown-gnu") for r in more_roots]+\
                             possible_roots

        if platform.machine() == arch:
            self.root = "/usr"
        else:
            for r in possible_roots:
                if os.path.exists(r) and os.path.isdir(r):
                    self.root = r
                    break

        self.cpuargs = cpuargs

        self.flags = [
                "-L",self.root,
                "-E",f"LD_LIBRARY_PATH={os.path.join(self.root,'lib')}:{os.path.join(self.root,'lib64')}",
                "-cpu",self.cpuargs
                ]

    def run(self, executable : str):
        log = logging.getLogger("EMULATOR")
        cmd = [self.exe]+self.flags+[executable]
        print(f"Command: {cmd}")
        process = Popen(cmd,
                        stdin=PIPE,stdout=PIPE,stderr=PIPE)
        process_out = process.communicate()
        output = process_out[0].decode()
        errout = process_out[1].decode()
        log.debug(f"Execution stdout: {output}")
        if 0 != process.returncode:
            log.error(f"Execution stderr: {errout}")
            raise RuntimeError(f"Error occurred running {executable}")
        log.debug(f"Execution stderr: {errout}")


# TODO: actual objects, check compatibility
emulators = {
        ("x86_64", "fma128")  : "qemu>7.2",
        ("x86_64", "fma256")  : "qemu>7.2",
        ("x86_64", "avx512")  : "sde",
        ("x86_64", "neon")    : "qemu",
        ("x86_64", "sve")     : "qemu",
        ("x86_64", "rvv")     : "qemu",
        ("aarch64", "fma128") : "qemu>7.2",
        ("aarch64", "fma256") : "qemu>7.2",
        ("aarch64", "neon")   : "qemu",
        ("aarch64", "sve")    : "qemu",
        ("aarch64", "rvv")    : "qemu",
        ("riscv64", "fma128") : "qemu>7.2",
        ("riscv64", "fma256") : "qemu>7.2",
        ("riscv64", "neon")   : "qemu",
        ("riscv64", "sve")    : "qemu",
        ("riscv64", "rvv")    : "vehave",
        ("riscv64", "rvv071")    : "vehave",
        }

def get_emulator(host_arch : str, target_isa : str):
    pass


emulation_commands = {
        "rvv071" : ["qemu-riscv64-static", "-L", "/usr/riscv64-linux-gnu",
                    "-cpu", "rv64,v=on,zba=on,vlen=512,vext_spec=v0.7.1"],
        "rvv" : ["qemu-riscv64-static", "-L", "/usr/riscv64-linux-gnu",
                 "-cpu", "rv64,v=on,zba=on,vlen=512,vext_spec=v1.0"],
        "neon" : ["qemu-aarch64-static", "-L", "/usr/aarch64-linux-gnu",
                  "-E","LD_LIBRARY_PATH=/usr/aarch64-linux-gnu/lib64"],
        "sve" : ["qemu-aarch64-static", "-L", "/usr/aarch64-linux-gnu",
                 "-E","LD_LIBRARY_PATH=/usr/aarch64-linux-gnu/lib64",
                 "-cpu", "max,sve=on,sve512=on"],
        "fma128" : ["sde", "-future", "--"],
        "fma256" : ["sde", "-future", "--"],
        "avx512" : ["sde", "-future", "--"],
        }
