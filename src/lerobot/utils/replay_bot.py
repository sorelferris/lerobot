import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from rich import print

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def _read_single_key() -> str:
    if sys.platform == "win32":
        import msvcrt

        return msvcrt.getch().decode("utf-8", errors="ignore")

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _wait_for_step_key() -> bool:
    print(
        "press [bold]space[/bold] to advance to the next step, press [bold]q[/bold] or [bold]ESC[/bold] to quit"
    )
    while True:
        key = _read_single_key()
        if key == " ":
            return True
        if key.lower() == "q" or key == "\x1b":
            return False


@dataclass
class ReplayRobotConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Index of the episode(s) to replay. Can be a single episode index or a comma-separated list of indices (e.g. "0,1,2").
    episode: str | None = None
    # Robot type
    type: str = "replay_robot"


class ReplayRobot:
    name = "replay_robot"

    def __init__(self, config: ReplayRobotConfig):
        self.config = config
        self.ds_meta = LeRobotDatasetMetadata(self.config.repo_id, root=self.config.root)
        self.camera_keys = self.ds_meta.camera_keys
        print(f"Total episodes in dataset {self.config.repo_id}: {self.ds_meta.total_episodes}")
        print(f"Camera keys: {self.ds_meta.camera_keys}")
        self.episodes = (
            [int(ep.strip()) for ep in self.config.episode.split(",")]
            if self.config.episode
            else list(range(self.ds_meta.total_episodes))
        )  # List of episodes to replay

        # Runtime variables
        self.dataset: LeRobotDataset | None = None
        self.frame_index = 0
        self._lock = threading.RLock()

    def _clone_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: self._clone_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._clone_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._clone_value(item) for item in value)
        return value

    def _get_current_item_locked(self):
        if self.dataset is None:
            return None

        if self.frame_index >= self.dataset.num_frames:
            return None

        return self.dataset[self.frame_index]

    def _get_current_item_snapshot(self):
        with self._lock:
            item = self._get_current_item_locked()
            if item is None:
                return None
            return {key: self._clone_value(value) for key, value in item.items()}

    def load_episode(self, episode_index: int):
        print(f"Loading episode {episode_index} from dataset {self.config.repo_id}...")
        dataset = LeRobotDataset(self.config.repo_id, root=self.config.root, episodes=[episode_index])
        with self._lock:
            self.dataset = dataset
            self.frame_index = 0

    def step(self):
        """Advance to the next frame in the episode."""
        with self._lock:
            if self.dataset is not None and self.frame_index < self.dataset.num_frames:
                self.frame_index += 1

    def describe_features(self):
        """Print the shapes and dtypes of the features in the current frame."""
        item = self._get_current_item_snapshot()
        if item is None:
            print("No episode loaded.")
            return

        with self._lock:
            current_frame_index = self.frame_index

        print(f"Features at frame index {current_frame_index}:")
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                if value.numel() > 0:
                    value_for_stats = (
                        value
                        if (torch.is_floating_point(value) or torch.is_complex(value))
                        else value.to(torch.float32)
                    )
                    mean = value_for_stats.mean().item()
                    _std = value_for_stats.std(unbiased=False).item()
                    _min = value_for_stats.min().item()
                    _max = value_for_stats.max().item()
                else:
                    mean = float("nan")
                    _std = float("nan")
                    _min = float("nan")
                    _max = float("nan")
                print(
                    f"  {key}: shape={value.shape}, dtype={value.dtype}, mean={mean:.4f}, std={_std:.4f}, min={_min:.4f}, max={_max:.4f}"
                )
            else:
                print(f"  {key}: type={type(value)}, value={value}")

    @property
    def teleop_action(self) -> torch.Tensor:
        """Get the teleoperation action at the current frame index."""
        item = self._get_current_item_snapshot()
        if item is None:
            raise RuntimeError("No episode loaded or episode is already finished.")
        return item["action"]

    @property
    def is_episode_done(self):
        with self._lock:
            return self.dataset is not None and self.frame_index >= self.dataset.num_frames

    def get_observation(self):
        """Get the observation at the current frame index."""
        item = self._get_current_item_snapshot()
        if item is None:
            return None

        return {
            "task": item["task"],
            "observation.state": item["observation.state"],
            **{key: item[key] for key in self.camera_keys},
        }

    def send_action(self, action_dict):
        self.step()  # Advance to the next frame (simulate sending action to the robot and moving to the next state)


def main(config: ReplayRobotConfig):
    print(asdict(config))
    robot = ReplayRobot(config)
    should_continue = True
    for episode_index in robot.episodes:
        if not should_continue:
            break
        robot.load_episode(episode_index)
        while not robot.is_episode_done:
            robot.describe_features()
            should_continue = _wait_for_step_key()
            if not should_continue:
                break
            time.sleep(0.1)  # Simulate time delay between actions


if __name__ == "__main__":
    main(tyro.cli(ReplayRobotConfig))
