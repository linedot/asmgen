"""
Calling convention abstraction class
"""

from typing import Callable, Union

from ..asmblocks.noarch import asmgen
from ..registers import asm_data_type as adt, greg_base, data_reg

class callconv:
    def __init__(self,
                 param_regs : dict[list[int]],
                 caller_save_lists : dict[list[int]],
                 callee_save_lists : dict[list[int]],
                 spreg : int
                 ):
        self.param_regs = param_regs
        self.caller_save_lists = caller_save_lists
        self.callee_save_lists = callee_save_lists
        self.spreg = spreg

        self.params = {}
        # stack offset for the next parameter
        self.spoff = 0
        # stack offset for save/restore
        self.spadd = 0
        # stack offset for different register types
        self.sp_rtype_offsets : dict[str,int] = {}

    def add_param(self, type_tag : str, name : str):

        if type_tag not in self.param_regs:
            raise ValueError(f"Invalid type tag: {type_tag}")

        if self.param_regs[type_tag]:
            idx = self.param_regs[type_tag].pop(0)
        else:
            idx = self.spoff
            type_tag = 'sp'
            self.spoff += 8 # TODO: non-64-bit


        self.params[name] = (type_tag, idx)

    def save_regs(self,
                  gen : asmgen,
                  reg_indices : dict[str,list[int]]) -> str:

        self.spadd = 8*sum([len(reg_list)\
            for _,reg_list in reg_indices.items()])

        spreg = gen.greg(self.spreg)
        asmblock = gen.add_greg_imm(
                reg=spreg,
                imm=-self.spadd
                )

        rtype_off = 0
        for type_tag,reg_list in reg_indices.items():
            store = getattr(gen,f"store_{type_tag}")
            self.sp_rtype_offsets[type_tag] = rtype_off
            kwargs = {}
            if 'freg' == type_tag:
                kwargs['dt'] = adt.FP64

            locations = self.get_locations(
                gen=gen,
                type_tag=type_tag,
                indices=reg_list)
            for i,reg in enumerate(locations):
                kwargs['areg'] = spreg
                kwargs['offset'] = 8*i+self.sp_rtype_offsets[type_tag]
                kwargs['src'] = reg
                asmblock += store(**kwargs)
                rtype_off += 8

        return asmblock

    def restore_regs(self,
                     gen : asmgen,
                     reg_indices : dict[str,list[int]]) -> str:

        spreg = gen.greg(self.spreg)

        asmblock = ""

        for type_tag,reg_list in reg_indices.items():
            load = getattr(gen,f"load_{type_tag}")

            kwargs = {}
            if 'freg' == type_tag:
                kwargs['dt'] = adt.FP64

            locations = self.get_locations(
                gen=gen,
                type_tag=type_tag,
                indices=reg_list)
            for i,reg in enumerate(locations):
                kwargs['areg'] = spreg
                kwargs['offset'] = 8*i+self.sp_rtype_offsets[type_tag]
                kwargs['dst'] = reg
                asmblock += load(**kwargs)

        asmblock += gen.add_greg_imm(
                reg=spreg,
                imm=self.spadd
                )

        self.spadd = 0

        return asmblock


    def get_locations(self,
                      gen : asmgen,
                      type_tag : str,
                      indices : list[int]) -> list[Union[greg_base,data_reg]]:

        kwargs = {
        }
        if 'freg' == type_tag:
            kwargs['dt'] = adt.FP64
        
        return [getattr(gen,type_tag)(**(kwargs|{'reg_idx':idx}))\
                for idx in indices]

        return locations

    def save_or_restore(self,
             gen : asmgen,
             regs : dict[list[int]],
             save_lists : dict[list[int]],
             action : Callable[[asmgen,dict[list[int]]], str]) -> str:

        reg_indices : dict[list[int]] = {
                k : [] for k in save_lists.keys()
                }

        for tag in save_lists.keys():
            if tag not in regs:
                continue
            reg_indices[tag] = set(regs[tag]).intersection(
                    set(save_lists[tag]))

        return action(gen=gen, reg_indices=reg_indices)

    def save_before_call(self,
                      gen : asmgen,
                      regs : dict[list[int]]) -> str:

        return self.save_or_restore(
                gen=gen,
                regs=regs,
                save_lists=self.caller_save_lists,
                action=self.save_regs)

    def restore_after_call(self,
                           gen : asmgen,
                           regs : dict[list[int]]) -> str:
        return self.save_or_restore(
                gen=gen,
                regs=regs,
                save_lists=self.caller_save_lists,
                action=self.restore_regs)

    def save_in_call(self,
                     gen : asmgen,
                     regs : dict[list[int]]) -> str:
        return self.save_or_restore(
                gen=gen,
                regs=regs,
                save_lists=self.callee_save_lists,
                action=self.save_regs)

    def restore_before_ret(self,
                     gen : asmgen,
                     used_gregs : list[int],
                     used_fregs : list[int]) -> str:
        return self.save_or_restore(
                gen=gen,
                regs=regs,
                save_lists=self.callee_save_lists,
                action=self.restore_regs)

    def get_params(self):
        return self.params

