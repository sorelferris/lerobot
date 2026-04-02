import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from rich import print

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


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
class ReplayBotConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Index of the episode(s) to replay. Can be a single episode index or a comma-separated list of indices (e.g. "0,1,2").
    episode: str | None = None  # If None, replay all episodes in the dataset.
    # Delay between interactive replay steps in seconds.
    step_delay_s: float = 0.1
    # Robot type
    type: str = "replay_bot"


@dataclass(slots=True)
class _FrameCache:
    dataset: LeRobotDataset | None = None
    frame_index: int | None = None
    item: dict | None = None
    observation: dict[str, torch.Tensor | str] | None = None


class ReplayBot:
    """A lightweight dataset-backed robot for deterministic episode replay.

    Only two pieces of mutable runtime state are tracked: the currently loaded
    episode dataset and the current frame index. They are guarded by a re-entrant
    lock so `capture_observation()` can safely be called from a background thread.
    """

    name = "replay_bot"

    def __init__(self, config: ReplayBotConfig):
        self.config = config
        self.ds_meta = LeRobotDatasetMetadata(config.repo_id, root=config.root)
        self.camera_keys = tuple(self.ds_meta.camera_keys)
        self.total_episodes = self.ds_meta.total_episodes
        self.episodes = self._parse_episodes(config.episode, self.ds_meta.total_episodes)

        self._lock = threading.RLock()
        self._dataset: LeRobotDataset | None = None  # Current episode dataset
        self._frame_index = 0  # Current frame index within the loaded episode
        self._frame_cache = _FrameCache()

        print(f"Total episodes in dataset {config.repo_id}: {self.ds_meta.total_episodes}")
        print(f"Camera keys: {self.camera_keys}")
        print(f"Feature keys: {self.ds_meta.features.keys()}")

    @staticmethod
    def _parse_episodes(episode_spec: str | None, total_episodes: int) -> list[int]:
        if not episode_spec:
            return list(range(total_episodes))

        episodes = [int(ep.strip()) for ep in episode_spec.split(",") if ep.strip()]
        if not episodes:
            raise ValueError("`episode` must contain at least one valid episode index.")
        return episodes

    @property
    def dataset(self) -> LeRobotDataset | None:
        with self._lock:
            return self._dataset

    @property
    def frame_index(self) -> int:
        with self._lock:
            return self._frame_index

    def _reset_cache_locked(self):
        self._frame_cache = _FrameCache()

    def _snapshot(self) -> tuple[int, dict] | None:
        """Return a cached snapshot of the current frame for thread-safe readers."""
        with self._lock:
            dataset = self._dataset
            frame_index = self._frame_index
            if dataset is None or frame_index >= dataset.num_frames:
                self._reset_cache_locked()
                return None

            cache = self._frame_cache
            if cache.dataset is dataset and cache.frame_index == frame_index and cache.item is not None:
                return frame_index, cache.item

            item = dict(dataset[frame_index].items())
            self._frame_cache = _FrameCache(dataset=dataset, frame_index=frame_index, item=item)
            return frame_index, item

    @staticmethod
    def _format_tensor_stats(value: torch.Tensor) -> str:
        if value.numel() == 0:
            return f"shape={value.shape}, dtype={value.dtype}, mean=nan, std=nan, min=nan, max=nan"

        value_for_stats = (
            value if (torch.is_floating_point(value) or torch.is_complex(value)) else value.to(torch.float32)
        )
        mean = value_for_stats.mean().item()
        std = value_for_stats.std(unbiased=False).item()
        min_value = value_for_stats.min().item()
        max_value = value_for_stats.max().item()
        return (
            f"shape={value.shape}, dtype={value.dtype}, mean={mean:.4f}, "
            f"std={std:.4f}, min={min_value:.4f}, max={max_value:.4f}"
        )

    def load_episode(self, episode_index: int):
        print(f"Loading episode {episode_index} from dataset {self.config.repo_id}...")
        dataset = LeRobotDataset(self.config.repo_id, root=self.config.root, episodes=[episode_index])
        with self._lock:
            self._dataset = dataset
            self._frame_index = 0
            self._reset_cache_locked()

    def step(self):
        """Advance to the next frame in the currently loaded episode."""
        with self._lock:
            dataset = self._dataset
            if dataset is not None and self._frame_index < dataset.num_frames:
                self._frame_index += 1
                self._reset_cache_locked()

    @property
    def is_episode_done(self) -> bool:
        with self._lock:
            dataset = self._dataset
            return dataset is not None and self._frame_index >= dataset.num_frames

    def describe_features(self):
        """Print the shapes and dtypes of the features in the current frame."""
        snapshot = self._snapshot()
        if snapshot is None:
            print("No episode loaded.")
            return

        frame_index, item = snapshot
        print(f"Features at frame index {frame_index}:")
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {self._format_tensor_stats(value)}")
            else:
                print(f"  {key}: type={type(value)}, value={value}")

    @property
    def teleop_action(self) -> torch.Tensor:
        """Return the recorded action for the current frame."""
        snapshot = self._snapshot()
        if snapshot is None:
            raise RuntimeError("No episode loaded or episode is already finished.")

        _, item = snapshot
        return item["action"]

    def capture_observation(self) -> dict[str, torch.Tensor | str] | None:
        """Return a thread-safe snapshot of the current observation."""
        with self._lock:
            cache_snapshot = self._snapshot()
            if cache_snapshot is None:
                return None

            _, item = cache_snapshot
            if self._frame_cache.observation is None:
                self._frame_cache.observation = {
                    "task": item["task"],
                    "observation.state": item["observation.state"],
                    **{key: item[key] for key in self.camera_keys},
                }

            return dict(self._frame_cache.observation)

    def get_observation(self) -> dict[str, torch.Tensor | str] | None:
        """Backward-compatible alias used by some replay utilities."""
        return self.capture_observation()

    def get_teleop_action(self) -> dict[str, torch.Tensor]:
        """Return the recorded action in the same shape as live robot APIs."""
        return {"action": self.teleop_action}

    def send_action(self, action: torch.Tensor):
        """Replay mode does not actuate hardware; keep the interface consistent."""
        return {"action": action}


def _replay_episode_interactively(robot: ReplayBot, episode_index: int, step_delay_s: float) -> bool:
    """Replay a single episode frame-by-frame until completion or user exit."""
    robot.load_episode(episode_index)
    print(f"[bold cyan]Replaying episode {episode_index}[/bold cyan]")

    while not robot.is_episode_done:
        robot.describe_features()
        if not _wait_for_step_key():
            print("[yellow]Replay interrupted by user.[/yellow]")
            return False

        if step_delay_s > 0:
            time.sleep(step_delay_s)
        robot.step()

    print(f"[green]Episode {episode_index} finished.[/green]")
    return True


def main(config: ReplayBotConfig):
    print(asdict(config))
    robot = ReplayBot(config)

    for episode_index in robot.episodes:
        if not _replay_episode_interactively(robot, episode_index, config.step_delay_s):
            break


if __name__ == "__main__":
    main(tyro.cli(ReplayBotConfig))
