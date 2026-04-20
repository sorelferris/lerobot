import pickle  # nosec
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from queue import Queue

import draccus
import torch
import zmq
from rich import print


class MockRobot:
    name = "mock_robot"

    def __init__(self, obs_features: dict[str, dict]):
        self.obs_features = obs_features
        self._zero_observation = {
            key: torch.zeros(
                feature["shape"],
                dtype=feature.get("dtype"),
                device=feature.get("device"),
            )
            for key, feature in self.obs_features.items()
        }

    def get_observation(self):
        # Return cached zero-filled observation for minimal overhead.
        return self._zero_observation

    def send_action(self, action_dict):
        pass


# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


@dataclass
class PolicyClientConfig:
    # Network configuration
    host: str = field(default="localhost", metadata={"help": "Server host"})
    port: int = field(default=8001, metadata={"help": "Server port"})

    def __post_init__(self):
        if not self.host:
            raise ValueError("host cannot be empty")

        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be in range [1, 65535], got {self.port}")


class PolicyClient:
    def __init__(self, config: PolicyClientConfig):
        self.config = config

        self.context = zmq.Context()
        self._socket = self.context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{self.config.host}:{self.config.port}")
        print(f"Connect to server: {self.config.host}:{self.config.port}")

        # Get policy name and config from server
        self.policy_name = self.request_policy_name()
        self.policy_config = self.request_policy_config()
        self.policy_repo_id = self.policy_config.get("repo_id", "N/A")
        self.input_features = self.policy_config.get("input_features", {})
        self.output_features = self.policy_config.get("output_features", {})
        print(f"[bright_yellow]Using policy: <{self.policy_name}>[/bright_yellow]")
        print(f"[bright_yellow]Policy repo_id: <{self.policy_repo_id}>[/bright_yellow]")
        print("Input features:")
        for key, value in self.input_features.items():
            print(f"  - {key}: {value}")
        print("Output features:")
        for key, value in self.output_features.items():
            print(f"  - {key}: {value}")

        self.timestep = 0  # Track the number of timesteps of action

    def _request(self, payload: dict, response_key: str, default=None):
        self._socket.send(pickle.dumps(payload))
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        start_time, end_time = time.time(), None
        while True:
            events = dict(poller.poll(timeout=1000))
            if self._socket in events:
                message = pickle.loads(self._socket.recv())
                if not message.get(response_key):
                    raise RuntimeError(f"Failed to get '{response_key}' from the policy server.")
                break
            else:
                elapsed = int(time.time() - start_time)
                print(
                    f"[bright_yellow]waiting for ZMQ server ({elapsed}s)...[/bright_yellow]",
                    end="\r",
                    flush=True,
                )
        if end_time is not None:
            print(f"[bright_yellow]Connected to server after {elapsed}s.[/bright_yellow]")
        return message.get(response_key, default)

    def request_policy_name(self):
        return self._request({"__request_policy_name__": True}, "policy_name")

    def request_policy_config(self):
        return self._request({"__request_policy_config__": True}, "policy_config")

    def update_observation(self, observation: dict) -> None:
        """Update the current observation from external code and enqueue it for the action request thread."""
        with self._last_observation_lock:
            self._last_observation = observation
        with self._obs_buffer_lock:
            self._obs_buffer.append(observation)
        self._obs_event.set()

    def close(self):
        self._socket.close(linger=0)
        self.context.term()
        print("[bright_yellow]PolicyClient closed[/bright_yellow]")

    def reset(self):
        self.timestep = 0  # Track the number of timesteps of action

    def request_actions(self):
        while not self.stop_event.is_set():
            # Wait for the latest observation provided via update_observation().
            if not self._obs_event.wait(timeout=0.1):
                continue

            with self._obs_buffer_lock:
                if not self._obs_buffer:
                    self._obs_event.clear()
                    continue
                observation = self._obs_buffer[-1]
                self._obs_buffer.clear()
                self._obs_event.clear()

            payload = {}
            # Include current timestep in the payload for better synchronization and debugging
            with self.timestep_lock:
                payload["timestep"] = self.timestep

            # Add timestamp to payload for debugging and latency measurement
            payload["timestamp"] = time.time()
            payload["observation"] = observation
            self._socket.send(pickle.dumps(payload))
            try:
                message = self._socket.recv()
            except zmq.error.Again:
                continue
            actions = pickle.loads(message)

            # Aggregate actions with the same timestep in the queue
            self.aggregate_actions_in_queue(actions, aggregate_fn=self.config.aggregate_fn)

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def require_action(self, observation: dict | None = None, timeout_s: float = 3.0) -> torch.Tensor | None:
        if observation is not None:
            self.update_observation(observation)
        try:
            action = self.action_queue.get(timeout=timeout_s)
        except Exception:
            print(
                f"[bright_red]Timeout requiring action after {timeout_s:.0f} seconds.[/bright_red]",
                end="\r",
                flush=True,
            )
            return None

        # Track queue size for debugging
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())

        # Increment timestep after getting action
        with self.timestep_lock:
            self.timestep += 1

        return action["action"]


@draccus.wrap()
def main(config: PolicyClientConfig):
    print(asdict(config))

    policy = PolicyClient(config)
    robot = MockRobot(obs_features=policy.input_features)

    try:
        dt = 1.0 / config.fps
        while True:
            step_start = time.perf_counter()
            observation = robot.get_observation()
            actions = policy.require_action(observation)
            if actions is not None:
                robot.send_action(actions)
            time.sleep(max(0.0, dt - (time.perf_counter() - step_start)))
            actual_fps = 1.0 / (time.perf_counter() - step_start)
            print(f"Actual FPS: {actual_fps:.2f}")

    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()
