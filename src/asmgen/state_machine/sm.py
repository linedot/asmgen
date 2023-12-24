import asmgen.state_machine.conditions as smconditions

class state:
    variables = {}
    def __init__(self):
        pass

class state_machine:

    states = {}
    variables = {}
    def __init__(self):
        pass

    def add_variable(self, name, default_value=0):
        if name == "":
            raise ValueError("Variable name can't be empty")
        self.variables[name] = default_value

    def add_state(self, name):
        if name == "":
            raise ValueError("State name can't be empty")
        self.states[name] = state()

    def add_transition_rule(self, src_state, dst_state, conditions):
        cant_msg = "Can't add transition rule:"
        assert src_state in self.states.keys(), f"{cant_msg} src_state \"{src_state}\" not in states list"
        assert dst_state in self.states.keys(), f"{cant_msg} dst_state \"{src_state}\" not in states list"
        for key in conditions.keys():
            assert key in self.variables, f"{cant_msg} variable {key}, for which conditions were defined, is not in variable list"
            condition = conditions[key]
            assert condition["type"] in smconditions.types
