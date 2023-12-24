def print_values(inputs, outputs, exprgen):
    inputs_str = "\n".join([f"std::cerr << \"{p.name} = \" << {p.name} << std::endl;" for i,p in enumerate(outputs)])
    outputs_str = "\n".join([f"std::cerr << \"{p.name} = \" << {p.name} << std::endl;" for i,p in enumerate(outputs)])
    return inputs_str+outputs_str

def equal_analyze(inputs, outputs, exprgen):
    return "\n".join([f"std::cerr << \"{p.name} must be \" << {exprgen('temps['+str({i})+']')} << \" but it's \" << {p.name} << std::endl;" for i,p in enumerate(outputs)])
