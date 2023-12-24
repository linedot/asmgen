from enum import Enum, unique
from typing import Union

headers_for_types = {
    "std::uint64_t" : "cstdint",
    "std::vector" : "vector"
}


@unique
class vio_type(Enum):
    INPUT = 1
    OUTPUT = 2

class vargen(object):

    def __init__(self):
        self.cppvars : list[str] = []
        self.cppvar_types : dict[str,str] = {}
        self.cppvar_inits : dict[str,str] = {}
        self.cppvar_vts : dict[str,vio_type] = {}
        self.required_headers : list[str] = []
        self.extra_decl_init : str = ""

    def new_var(self, cpp_type : str, name : str ="#auto#",
                vt : vio_type =vio_type.OUTPUT) -> str:
        if "#auto#" == name:
            name = f"tmpvar{len(self.cppvars)}"
        base_type = cpp_type
        if "<" in cpp_type:
            base_type = cpp_type.split("<")[0]
        if base_type in headers_for_types:
            self.required_headers.append(headers_for_types[base_type])
        self.cppvars.append(name)
        self.cppvar_types[name] = cpp_type
        self.cppvar_inits[name] = f"{name}"
        self.cppvar_vts[name] = vt
        return name

    def new_vector(self, cpp_type : str, size : Union[str,int]=1,
                   fillwith : Union[int,float]=0.0,
                   name : str ="#auto#",
                   vt : vio_type =vio_type.OUTPUT) -> str:
        if "#auto#" == name:
            name = f"tmpvar{len(self.cppvars)}"
        base_type = cpp_type
        if base_type in headers_for_types:
            self.required_headers.append(headers_for_types[base_type])
        if not "vector" in self.required_headers:
            self.required_headers.append("vector")
        self.cppvars.append(name)
        self.cppvar_types[name] = f"std::vector<{cpp_type}>"
        self.cppvar_inits[name] = f"{name}ptr"
        self.cppvar_vts[name] = vt

        self.extra_decl_init += f"{name}.resize({size});\n" 
        self.extra_decl_init += f"std::fill({name}.begin(),{name}.end(),{fillwith});\n" 
        self.extra_decl_init += f"void* {name}ptr = {name}.data();\n"
        
        return name

    def custom_var_init(self, name : str, init : str):
        self.cppvar_inits[name] = init

    def get_declarations(self) -> str:
        declarations = "\n".join([f"{self.cppvar_types[name]} {name};" for name in self.cppvars])
        declarations += "\n"
        declarations += self.extra_decl_init
        return declarations

    def get_includes(self) -> str:
        includes = "\n".join([f"#include <{header}>" for header in self.required_headers])
        return includes

    def get_variables(self) -> list[tuple[str,str,vio_type]]:
        return [(var,self.cppvar_inits[var],self.cppvar_vts[var]) for var in self.cppvars]

    def get_var_inits(self) -> dict[str,str]:
        return self.cppvar_inits

    def reset_variables(self):
        self.cppvars = []
        self.cppvar_types = {}
        self.cppvar_inits = {}
        self.cppvar_vts = {}
        self.extra_decl_init = ""
