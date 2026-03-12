import logging

from .callconv import callconv
from ..asmblocks.noarch import asmgen
from ..registers import reg_tracker

class fngen:

    def __init__(self, gen : asmgen, rt : reg_tracker):
        self.gen = gen
        self.rt = rt

        self.required_loads : dict[str,int] = dict()

        self.debug_block : str = ""

        self.debug = logging.getLogger("FNGEN").debug

    def init_cc(self, cc : callconv,
                reverse_alias_map : dict[str,str] = dict(),
                unused_parameters : set[str] = set()):

        params = cc.get_params()
        for name,(tag,idx) in params.items():
            if name in reverse_alias_map:
                name = reverse_alias_map[name]
            if name in unused_parameters:
                continue
            if tag in ['greg','freg']:
                self.rt.reserve_specific_reg(tag, idx)
                self.rt.alias_reg(tag, name, idx)

        for name,(tag,idx) in params.items():
            if name in reverse_alias_map:
                name = reverse_alias_map[name]
            if name in unused_parameters:
                continue
            if tag == 'sp':
                regidx = self.rt.reserve_any_reg('greg')
                self.rt.alias_reg('greg', name, regidx)
                self.required_loads[name] = idx

        self.debug_block = ""
        for name in params.keys():
            if name in reverse_alias_map:
                name = reverse_alias_map[name]
            if name in unused_parameters:
                continue
            reg = self.gen.greg(self.rt.aliased_regs['greg'][name])
            self.debug_block += self.gen.asmwrap(f"# {reg} = {name}")
            self.debug(f"Allocated {reg} for {name}")

    def get_boilerplate(self,
                        cc : callconv,
                        unused_parameters : set[str] = set()):
        used_gregs = self.rt.get_clobbered_regs('greg')
        used_fregs = self.rt.get_clobbered_regs('freg')

        if self.gen.are_fregs_in_vregs:
            used_vregs = self.rt.get_clobbered_regs('vreg')
            used_fregs = list(set(used_fregs).union(set(used_vregs)))

        sr_count = len(set(used_gregs).intersection(cc.callee_save_lists['greg']))
        sr_count += len(set(used_fregs).intersection(cc.callee_save_lists['freg']))
        saveblock = cc.save_in_call(self.gen, 
                                    regs={'greg':used_gregs,
                                          'freg':used_fregs})

        loadblock = ''

        for name,off in self.required_loads.items():
            loadblock += self.gen.asmwrap(f"# loading {name}")
            loadblock += self.gen.load_greg(
                    areg=self.gen.greg(cc.spreg),
                    offset=off+sr_count*8, # Assuming 64 bit/ 8 byte pointers
                    dst=self.gen.greg(self.rt.aliased_regs['greg'][name])
            )


        restoreblock = cc.restore_before_ret(self.gen,
                                             regs={'greg':used_gregs,
                                                   'freg':used_fregs})

        return saveblock,self.debug_block+loadblock,restoreblock
