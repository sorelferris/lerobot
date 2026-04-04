import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from queue import Queue

import draccus
import torch
import zmq
from rich import print


class MockRobot:
    name = "mock_robot"

    def __init__(self, config):
        self.config = config

    def get_observation(self):
        # Return dummy observation in torch tensor format
        return {
            "task": "do nothing",
            "observation.state": torch.randn(6),  # 8-dimensional state vector
            "observation.images.front": torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8).float(),
            "observation.images.wrist": torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8).float(),
        }

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

    # Runtime configuration
    fps: int = field(default=30, metadata={"help": "Frames per second"})

    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Name of aggregate function to use. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )

    def __post_init__(self):
        if not self.host:
            raise ValueError("host cannot be empty")

        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be in range [1, 65535], got {self.port}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.aggregate_fn_name not in AGGREGATE_FUNCTIONS:
            available = list(AGGREGATE_FUNCTIONS.keys())
            raise ValueError(f"Unknown aggregate function '{self.aggregate_fn_name}'. Available: {available}")
        self.aggregate_fn = AGGREGATE_FUNCTIONS[self.aggregate_fn_name]


class PolicyClient:
    def __init__(self, config: PolicyClientConfig, obs_fn: Callable):
        self.config = config
        assert callable(obs_fn), "obs_fn must be a callable function"
        self.obs_fn = obs_fn

        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REQ)
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

        # Initialize client side variables
        self.timestep = 0  # Track the number of timesteps of action
        self.timestep_lock = threading.Lock()  # Protect timestep variable
        self._last_observation = None
        self._last_observation_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.action_thread = threading.Thread(target=self.request_actions, daemon=True)
        self.action_thread.start()
        print("[bright_yellow]Client started and action thread launched[/bright_yellow]")

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

    def stop(self):
        self.stop_event.set()
        self.action_thread.join()
        self._socket.close(linger=0)
        self._ctx.term()
        print("[bright_yellow]Client stopped[/bright_yellow]")

    def reset(self):
        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.timestep = 0  # Track the number of timesteps of action
        self.timestep_lock = threading.Lock()  # Protect timestep variable
        with self._last_observation_lock:
            self._last_observation = None

    @property
    def last_observation(self) -> dict | None:
        with self._last_observation_lock:
            return self._last_observation

    def aggregate_actions_in_queue(self, incoming_actions: dict, aggregate_fn: Callable = None):
        """Aggregate actions in the queue"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue

        current_action_queue = {x["timestep"]: x["action"] for x in internal_queue.queue}

        for timestep, action in incoming_actions.items():
            with self.timestep_lock:
                current_timestep = self.timestep

            # Skip actions that are already passed
            if timestep <= current_timestep:
                continue

            # Add action with new timestep
            elif timestep not in current_action_queue:
                future_action_queue.put({"timestep": timestep, "action": action})
                continue

            # Aggregate action with the same timestep in the queue
            future_action_queue.put(
                {"timestep": timestep, "action": aggregate_fn(current_action_queue[timestep], action)}
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def request_actions(self):
        while not self.stop_event.is_set():
            # Capture observation and send to policy server
            observation = self.obs_fn()
            if observation is None:
                time.sleep(0.01)
                continue

            with self._last_observation_lock:
                self._last_observation = observation

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

    def require_action(self, timeout_s: float = 10.0) -> torch.Tensor | None:
        start_time = time.monotonic()
        while not self.actions_available():
            if time.monotonic() - start_time >= timeout_s:
                print(f"[bright_red]Timeout waiting for action after {timeout_s} seconds.[/bright_red]")
                return None
            time.sleep(0.01)  # Wait until action is available

        # Get action from queue safely and track queue size for debugging
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            action = self.action_queue.get_nowait()

        # Increment timestep after getting action
        with self.timestep_lock:
            self.timestep += 1

        return action["action"]


@draccus.wrap()
def async_client_zmq(config: PolicyClientConfig):
    print(asdict(config))

    robot = MockRobot(config)
    policy = PolicyClient(config, robot.get_observation)

    try:
        dt = 1.0 / config.fps
        while True:
            step_start = time.perf_counter()
            actions = policy.require_action()
            if actions is not None:
                robot.send_action(actions)
            time.sleep(max(0.0, dt - (time.perf_counter() - step_start)))
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        policy.stop()


if __name__ == "__main__":
    async_client_zmq()
