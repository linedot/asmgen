# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Analysers for outcomes of different condition checks
"""

def print_values(inputs, outputs):
    """
    prints inputs and outpus
    """
    inputs_str = "\n".join([(f"std::cerr << \"{p.name} = \""
                             f" << {p.name} << std::endl;") \
                             for p in inputs])
    outputs_str = "\n".join([(f"std::cerr << \"{p.name} = \""
                             f" << {p.name} << std::endl;") \
                             for p in outputs])
    return inputs_str+"\n"+outputs_str

def equal_analyze(outputs, exprgen):
    """
    Analysis for an equal comparison
    """
    return "\n".join([(f"std::cerr << \"{p.name} must be \" "
                       f" << {exprgen('temps['+str({i})+']')}"
                       f" << \" but it's \" << {p.name} << std::endl;") \
                       for i,p in enumerate(outputs)])
