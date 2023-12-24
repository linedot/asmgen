from asmgen.compilation.compiler_presets import default_flags, output_flag, stdin_flags, lib_flags, arch_flags, cross_cxx_flags, cross_lib_flags

from subprocess import Popen, PIPE

import logging

class compiler(object):
    def __init__(self, executable, arch):
        self.executable = executable
        if self.executable in default_flags:
            self.flags = default_flags[self.executable]
        if self.executable in output_flag:
            self.oflag = output_flag[self.executable]
        if self.executable in lib_flags:
            self.lib_flags = lib_flags[self.executable]
        if self.executable in stdin_flags:
            self.stdin_flags = stdin_flags[self.executable]
        if self.executable in arch_flags:
            self.arch_flags = arch_flags[self.executable][arch]

    def compile_lib(self, source, output_filename, cross_compile="native"):
        log = logging.getLogger("COMPILATION")
        cross_flags = []
        if "native" != cross_compile:
            cross_flags = cross_cxx_flags[cross_compile][self.executable] +\
                          cross_lib_flags[cross_compile][self.executable]
        cmd = [self.executable] +\
               self.flags +\
               self.arch_flags +\
               self.lib_flags +\
               cross_flags +\
               [self.oflag,output_filename] +\
               self.stdin_flags

        log.debug(f"Compiling c++ code for {output_filename} using:")
        log.debug(f"{' '.join(cmd)}")
        log.debug(f"...")
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        process_out = p.communicate(input=source.encode())
        output = process_out[0].decode()
        errout = process_out[1].decode()
        log.debug(f"Compilation stdout: {output}")
        log.debug(f"Compilation stderr: {errout}")

        return 0 == p.returncode

    def compile_exe(self, source, output_filename, libs, cross_compile="native"):
        log = logging.getLogger("COMPILATION")
        cross_flags = []
        if "native" != cross_compile:
            cross_flags = cross_cxx_flags[cross_compile][self.executable] +\
                          cross_lib_flags[cross_compile][self.executable]
        cmd = [self.executable] +\
               self.flags +\
               self.arch_flags +\
               cross_flags +\
               [self.oflag,output_filename] +\
               libs +\
               self.stdin_flags

        log.debug(f"Compiling c++ code for {output_filename} using:")
        log.debug(f"{' '.join(cmd)}")
        log.debug(f"...")
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        process_out = p.communicate(input=source.encode())
        output = process_out[0].decode()
        errout = process_out[1].decode()
        log.debug(f"Compilation stdout: {output}")
        log.debug(f"Compilation stderr: {errout}")

        return 0 == p.returncode
