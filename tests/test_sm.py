from asmgen.state_machine.sm import state_machine, state

import unittest

class state_machine_test(unittest.TestCase):

    def test_add_state(self):
        sm = state_machine()
        sm.add_state("state1")

    def test_add_state2(self):
        sm = state_machine()
        sm.add_state("state1")
        sm.add_state("state2")

    def test_add_variable(self):
        sm = state_machine()
        sm.add_variable("x")

    def test_add_state_and_variable(self):
        sm = state_machine()
        sm.add_state("state1")
        sm.add_variable("x")

    def test_assert_src_state(self):
        sm = state_machine()
        sm.add_state("state1")

        with self.assertRaises(AssertionError):
            sm.add_transition_rule("invalid","state1",{})

    def test_assert_dst_state(self):
        sm = state_machine()
        sm.add_state("state1")

        with self.assertRaises(AssertionError):
            sm.add_transition_rule("state1","invalid",{})

    def test_assert_empty_state_name(self):
        sm = state_machine()
        with self.assertRaises(ValueError):
            sm.add_state("")

    def test_assert_empty_variable_name(self):
        sm = state_machine()
        with self.assertRaises(ValueError):
            sm.add_variable("")

    def test_assert_cond_key_variables(self):
        sm = state_machine()
        sm.add_state("state1")
        sm.add_state("state2")

        sm.add_variable("x", 0)

        conditions = {"y": {"type" : "equal", "value": 0}}

        with self.assertRaises(AssertionError):
            sm.add_transition_rule("state1", "state2", conditions)

    def test_assert_cond_invalid_type(self):
        sm = state_machine()
        sm.add_state("state1")
        sm.add_state("state2")

        sm.add_variable("x", 0)

        conditions = {"x": {"type" : "foobar", "value": 0}}

        with self.assertRaises(AssertionError):
            sm.add_transition_rule("state1", "state2", conditions)
    

def main():
    unittest.main()


if "__main__" == __name__:
    main()
