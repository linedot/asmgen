import asmgen.cppgen.templates as cpptpl
from mako.template import Template


def write_test_asmblock_func(name, asmblock, tparams):
    template = cpptpl.test_func
    func_source = Template(template).render(
            asmblock=asmblock,
            function_name=name,
            function_params=tparams["function_params"],
            header=tparams["header"],
            prepare=tparams["prepare"],
            checkfun_definition=tparams["checkfun_definition"],
            check=tparams["check"],
            analyze=tparams["analyze"])
    return func_source+"\n\n"

def write_test_func_declaration(name, tparams):
    template = cpptpl.test_func_declaration
    func_source = Template(template).render(
            function_name=name,
            function_params=tparams["function_params"])
    return func_source+"\n\n"

def write_asmblock_func(name, asmblock, tparams):
    template = cpptpl.simple_func
    func_source = Template(template).render(
            asmblock=asmblock,
            function_name=name,
            function_params=tparams["function_params"],
            prepare=tparams["prepare"])
    return func_source+"\n\n"

def write_standalone_asmblock_test(asmblock, tparams):
    template = cpptpl.simple_test
    func_source = Template(template).render(
            asmblock=asmblock,
            header=tparams["header"],
            checkfun_definition=tparams["checkfun_definition"],
            tmp_decl=tparams["tmp_decl"],
            tmp_assign=tparams["tmp_assign"],
            inparam_definition=tparams["inparam_definition"],
            outparam_definition=tparams["outparam_definition"],
            check=tparams["check"],
            analyze=tparams["analyze"])
    return func_source+"\n\n"
