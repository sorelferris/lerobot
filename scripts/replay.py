import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tyro
from rich import print
from rich.progress import Progress

from lerobot.utils.replay_bot import ReplayRobot, ReplayRobotConfig
from lerobot.utils.rerun_utils import RerunLogger
from lerobot.async_inference.robot_client_zmq import PolicyClient, PolicyClientConfig


def pick_camera(observation: dict[str, any], token: str):
    """Pick the first camera image whose key contains the token."""
    image_keys = [k for k in observation if "observation.images" in k]
    for key in image_keys:
        if token in key.lower():
            return observation[key]
    return None


@dataclass
class ReplayConfig:
    # Policy configuration
    policy: PolicyClientConfig
    # Robot configuration
    robot: ReplayRobotConfig
    # Limit the frames per second. By default, uses the dataset fps.
    fps: int | None = None
    # Directory to save action data for comparison
    save_dir: str = "outputs/replay"
    # Optional: Remote Rerun viewer URL for visualization. If None, no visualization.
    rerun_url: str | None = None


def replay(config: ReplayConfig):
    print(asdict(config))

    robot = ReplayRobot(config.robot)
    policy = PolicyClient(config.policy, get_observation=robot.get_observation)

    # Create output directory if it doesn't exist
    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    repo_name = config.robot.repo_id.replace("/", "_")

    # Initialize RerunLogger
    logger = RerunLogger(url=config.rerun_url)

    try:
        for episode in robot.episodes:
            robot.load_episode(episode)

            # Reset policy and replay episode
            policy.reset()

            policy_actions = []
            teleop_actions = []
            frame_count = 0

            logger.switch_record()
            # Replay episode until done with MockRobot providing observations and PolicyClient providing actions
            with Progress() as progress:
                task_id = progress.add_task("Replaying episode", total=robot.dataset.num_frames)
                while not robot.is_episode_done:
                    start = time.perf_counter()

                    # Get action from policy
                    action = policy.require_action()

                    if action is None:
                        # Fallback to teleop action if policy doesn't provide an action in time
                        action = robot.teleop_action

                    # Store actions for later comparison
                    teleop_actions.append(robot.teleop_action.numpy())
                    policy_actions.append(action.numpy())

                    # Log to Rerun
                    observation = robot.get_observation()
                    head = pick_camera(observation, "head")
                    left = pick_camera(observation, "left")
                    right = pick_camera(observation, "right")

                    if head is not None:
                        data = {
                            "observation.images.head_camera": head,
                            "observation.state": observation["observation.state"],
                            "teleop_action": robot.teleop_action,
                            "policy_action": action,
                            "framestep": frame_count,
                        }
                        if left is not None:
                            data["observation.images.left_camera"] = left
                        if right is not None:
                            data["observation.images.right_camera"] = right
                        logger.log(data)

                    # Send action to the robot (which will advance to the next frame)
                    robot.send_action(action)
                    frame_count += 1

                    # Busy-wait to maintain the desired fps
                    time.sleep(max(0.0, 1.0 / config.fps - (time.perf_counter() - start)))

                    frame_interval = time.perf_counter() - start
                    real_fps = 1.0 / frame_interval if frame_interval > 0 else float("inf")

                    progress.update(
                        task_id,
                        advance=1,
                        description=f"Replaying {len(policy_actions)}/{robot.dataset.num_frames} ({frame_interval * 1000:.2f} ms, {real_fps:.2f} fps)",
                    )

            policy_actions = np.array(policy_actions)
            teleop_actions = np.array(teleop_actions)
            print(f"{teleop_actions.shape=}, {policy_actions.shape=}, {robot.dataset.num_frames=}")

            # Save action data for later comparison
            np.save(save_path / f"teleop_actions_{repo_name}_episode_{episode}_tel.npy", teleop_actions)
            np.save(
                save_path / f"policy_actions_{repo_name}_episode_{episode}_{policy.policy_name}_{config.policy.aggregate_fn_name}.npy",
                policy_actions,
            )

    finally:
        if logger:
            logger.stop()
        policy.stop()


if __name__ == "__main__":
    replay(tyro.cli(ReplayConfig))
