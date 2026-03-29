# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example:
```shell
python -m lerobot.async_inference.policy_server_zmq \
    --host=127.0.0.1 \
    --port=5555 \
    --policy_type=act \
    --pretrained_name_or_path=/path/to/checkpoint \
    --policy_device=cuda \
    --fps=30 \
    --inference_latency=0.033
```
"""

import pickle  # nosec
import threading
import time
from dataclasses import asdict, dataclass, field

import draccus
import torch
import zmq
from rich import print
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from lerobot.policies.factory import get_policy_class, make_pre_post_processors


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8000, metadata={"help": "Port number to bind the server to"})

    # Policy configuration
    policy_type: str = field(default="act", metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str = field(default="", metadata={"help": "Pretrained model name or path"})
    policy_device: str = field(default="cuda", metadata={"help": "Device for policy inference"})
    rename_map: dict[str, str] = field(default_factory=dict)

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int = field(default=50, metadata={"help": "Number of actions per chunk"})

    # Timing configuration
    fps: int = field(default=30, metadata={"help": "Frames per second"})
    inference_latency: float = field(default=0.033, metadata={"help": "Target inference latency in seconds"})

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")

        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")


class PolicyServer:
    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.actions_per_chunk = config.actions_per_chunk
        self.policy = self._setup_policy()
        self.chunk_size = 0
        self.action_dim = 0
        device_override = {"device": config.policy_device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=config.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": config.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )
        print("preprocessor steps:")
        for idx, step in enumerate(self.preprocessor.steps):
            device = getattr(step, "device", None)
            print(f"  [{idx}] {step.__class__.__name__} {'device=' + str(device) if device else ''}")
        print("postprocessor steps:")
        for idx, step in enumerate(self.postprocessor.steps):
            device = getattr(step, "device", None)
            print(f"  [{idx}] {step.__class__.__name__} {'device=' + str(device) if device else ''}")

        self._warmup_policy()

        self.stop_event = threading.Event()
        self.actions_per_chunk = config.actions_per_chunk
        self.last_processed_obs = None
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{config.host}:{config.port}")
        self.console = Console()

    def _setup_policy(self):
        policy_class = get_policy_class(self.config.policy_type)
        print(f"Load <{policy_class.__name__}> policy from {self.config.pretrained_name_or_path}")
        start = time.perf_counter()
        policy = policy_class.from_pretrained(self.config.pretrained_name_or_path)
        policy.to(self.config.policy_device)
        elapsed = time.perf_counter() - start
        device = self.config.policy_device
        print(policy.config)
        print(f"[bright_yellow]Taken {elapsed:.2f} seconds to put policy on {device}.[/bright_yellow]")
        return policy

    def _warmup_policy(self):
        """Warm up the policy by running dummy inferences."""
        start_time = time.monotonic()
        for _ in range(3):
            dummy_observation = {
                "observation.state": torch.rand(8),
                "observation.images.head_camera": torch.rand(3, 480, 640),
                "observation.images.left_camera": torch.rand(3, 480, 640),
                "task": "do something",
            }
            self.predict_action_chunk(dummy_observation, i0=0)
        elapsed = time.perf_counter() - start_time
        print(f"[bright_yellow]Taken {elapsed:.2f} seconds to warm up the policy.[/bright_yellow]")

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    @property
    def policy_config(self):
        return asdict(self.policy.config)

    def predict_action_chunk(self, observation: dict[str, torch.Tensor], i0: int):
        # Check observation dict values for torch.Tensor type
        for k, v in observation.items():
            if k == "task":
                continue
            assert isinstance(v, torch.Tensor), (
                f"Observation '{k}' must be a torch.Tensor, got {type(v).__name__}"
            )

        self.last_processed_obs = observation

        start_time = time.perf_counter()
        observation = self.preprocessor(observation)

        action_tensor = self.policy.predict_action_chunk(observation)
        action_tensor = action_tensor[:, : self.actions_per_chunk, :]  # slice actions_per_chunk

        _, self.chunk_size, self.action_dim = action_tensor.shape  # (B, chunk_size, action_dim)

        # Process each action in the chunk
        processed_actions = []
        for i in range(self.chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        action_tensor = action_tensor.detach().cpu()

        # Convert to dict with timestep keys
        actions = {i0 + i: action for i, action in enumerate(action_tensor)}

        elapsed_time = time.perf_counter() - start_time

        # If inference is faster than target latency, artificially delay to maintain consistent timing
        if elapsed_time < self.config.inference_latency:
            time.sleep(self.config.inference_latency - elapsed_time)

        return actions

    def run(self) -> None:
        print(f"Policy server listen on: tcp://{self.config.host}:{self.config.port}")
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        last_step = 0
        try:
            with Live(console=self.console, refresh_per_second=4) as live:
                while self.stop_event.is_set() is False:
                    delay_time = time.perf_counter()
                    events = dict(poller.poll(timeout=200))
                    delay_time = (time.perf_counter() - delay_time) * 1000  # convert to ms

                    if self.socket not in events:
                        continue

                    payload = self.socket.recv()

                    unpac_time = time.perf_counter()
                    message = pickle.loads(payload)
                    unpac_time = (time.perf_counter() - unpac_time) * 1000  # convert to ms

                    if not isinstance(message, dict):
                        print(f"Invalid message format: expected dict, got {type(message).__name__}")
                        self.socket.send(b"")
                        continue

                    if message.get("__request_policy_name__", False):
                        self.socket.send(pickle.dumps({"policy_name": self.policy.__class__.__name__}))
                        continue

                    observation = message.get("observation")
                    timestamp = message.get("timestamp", time.time())
                    timestep = message.get("timestep", 0)

                    infer_time = time.perf_counter()
                    actions = self.predict_action_chunk(observation, i0=timestep)
                    self.socket.send(pickle.dumps(actions))
                    infer_time = (time.perf_counter() - infer_time) * 1000  # convert to ms

                    # Update live panel
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

                    table = Table(show_header=False, box=None, padding=(0, 1))
                    table.add_column(style="bold cyan", no_wrap=True)
                    table.add_column(style="white", no_wrap=True)
                    table.add_column(style="bold cyan", no_wrap=True)
                    table.add_column(style="white", no_wrap=True)

                    skip = timestep - last_step
                    table.add_row("time steps", f"{timestep} (+{skip})", "delay_time", f"{delay_time:.2f} ms")
                    last_step = timestep
                    table.add_row("chunk_size", str(self.chunk_size), "unpac_time", f"{unpac_time:.2f} ms")
                    table.add_row("action_dim", str(self.action_dim), "infer_time", f"{infer_time:.2f} ms")
                    table.add_row("", "", "", "")
                    for k, v in observation.items():
                        val = (
                            f"{tuple(v.shape)} [{v.min():.2f}, {v.max():.2f}]"
                            if isinstance(v, torch.Tensor)
                            else v
                        )
                        table.add_row(k, val, "", "")

                    task = observation.get("task", "Unknown Task")
                    panel = Panel(table, title=f"{self.policy.__class__.__name__}: {task}", subtitle=f"{timestamp}")
                    live.update(panel)
        except KeyboardInterrupt:
            print("\n[bright_yellow]Received Ctrl+C, shutting down policy server...[/bright_yellow]")
        finally:
            self.stop_event.set()
            poller.unregister(self.socket)
            self.socket.close(linger=0)
            self.context.term()
            print("Policy Server terminated")


@draccus.wrap()
def serve_policy(config: PolicyServerConfig):
    print(asdict(config))
    server = PolicyServer(config)
    server.run()


if __name__ == "__main__":
    serve_policy()
