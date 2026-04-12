import threading
import time
import unittest

from run_robots import RobotExecutionState, RobotThread
from robot_common.robot_api import BaseRobotController, RobotState


class DummyController(BaseRobotController):
    def get_robot_state(self):
        return RobotState(timestamp=self.data.time)


class FakeData:
    def __init__(self):
        self.time = 0.0


class RobotExecutionStateTests(unittest.TestCase):
    def test_validate_wait_target_accepts_known_robot_and_index(self):
        state = RobotExecutionState({"belt": 5, "arm2": 9})
        state.validate_wait_target("arm2", 7)

    def test_validate_wait_target_rejects_unknown_robot(self):
        state = RobotExecutionState({"belt": 5})
        with self.assertRaisesRegex(ValueError, "unknown target robot"):
            state.validate_wait_target("arm2", 0)

    def test_validate_wait_target_rejects_out_of_range_index(self):
        state = RobotExecutionState({"arm2": 3})
        with self.assertRaisesRegex(ValueError, "out of range"):
            state.validate_wait_target("arm2", 3)


class WaitActionTests(unittest.TestCase):
    def test_wait_returns_after_target_action_completion(self):
        data = FakeData()
        state = RobotExecutionState({"belt": 3, "arm2": 4})
        controller = DummyController(model=None, data=data, robot_name="belt")
        controller.set_execution_state(state)

        def complete_target():
            time.sleep(0.05)
            state.mark_action_complete("arm2", 1)

        thread = threading.Thread(target=complete_target, daemon=True)
        thread.start()
        controller.action_wait(robot="arm2", action_index=1)
        self.assertTrue(state.is_action_complete("arm2", 1))

    def test_wait_rejects_invalid_target_before_blocking(self):
        data = FakeData()
        state = RobotExecutionState({"belt": 3})
        controller = DummyController(model=None, data=data, robot_name="belt")
        controller.set_execution_state(state)

        with self.assertRaisesRegex(ValueError, "unknown target robot"):
            controller.action_wait(robot="arm2", action_index=0)


class RobotThreadStateTests(unittest.TestCase):
    def test_thread_marks_actions_started_and_completed(self):
        data = FakeData()
        controller = DummyController(model=None, data=data, robot_name="arm2")
        state = RobotExecutionState({"arm2": 2})
        controller.set_execution_state(state)
        sequence = [
            {"action": "idle", "parameters": {"duration": 0.5}},
            {"action": "idle", "parameters": {"duration": 0.5}},
        ]
        thread = RobotThread(controller, sequence, state)

        thread.start()
        while thread.running:
            data.time += 0.05
            time.sleep(0.001)
        thread.join(timeout=1.0)

        self.assertTrue(state.is_action_complete("arm2", 0))
        self.assertTrue(state.is_action_complete("arm2", 1))
        self.assertEqual(state.current_action_index["arm2"], 1)


class WaitSequencingTests(unittest.TestCase):
    def test_wait_unblocks_only_after_target_action_finishes(self):
        data = FakeData()
        state = RobotExecutionState({"belt": 2, "arm2": 2})
        belt = DummyController(model=None, data=data, robot_name="belt")
        arm2 = DummyController(model=None, data=data, robot_name="arm2")
        belt.set_execution_state(state)
        arm2.set_execution_state(state)

        belt_sequence = [
            {"action": "idle", "parameters": {"duration": 0.5}},
            {"action": "wait", "parameters": {"robot": "arm2", "action_index": 0}},
        ]
        arm2_sequence = [
            {"action": "wait", "parameters": {"robot": "belt", "action_index": 0}},
            {"action": "idle", "parameters": {"duration": 0.5}},
        ]

        belt_thread = RobotThread(belt, belt_sequence, state)
        arm_thread = RobotThread(arm2, arm2_sequence, state)
        belt_thread.start()
        arm_thread.start()

        deadline = time.time() + 2.0
        while (belt_thread.running or arm_thread.running) and time.time() < deadline:
            data.time += 0.05
            time.sleep(0.001)

        belt_thread.join(timeout=1.0)
        arm_thread.join(timeout=1.0)
        self.assertTrue(state.is_action_complete("belt", 1))
        self.assertTrue(state.is_action_complete("arm2", 1))


if __name__ == "__main__":
    unittest.main()
