import time
from dataclasses import asdict, dataclass

import numpy as np
import tyro  # pip install tryo
from rich import print  # pip install rich

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.robots.ros_robot import RosRobot, RosRobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.rerun_utils import RerunLogger
from lerobot.scripts.client import PolicyClient, PolicyClientConfig


@dataclass
class PlayConfig:
    # Task string sent to the policy (e.g. "pick up the bin")
    task: str
    # Limit the frames per second. By default, uses the policy fps.
    fps: int
    # Robot configuration
    robot: RosRobotConfig
    # Policy configuration
    policy: PolicyClientConfig
    # Optional: Remote Rerun viewer URL for visualization. If None, no visualization.
    rerun_url: str | None = None


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {
        "play": False,
        "stop": False,
        "forward": False,
        "policy_control": True,
    }

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.space:
                events["play"] = not events["play"]
                print(f"[yellow]Toggled play: {'▶️' if events['play'] else '⏸️'}[/yellow]")
            elif key == keyboard.Key.esc:
                events["stop"] = True
                print("[red]Pressed [ESC]. Bye![/red]")
            elif key == keyboard.Key.right:
                events["forward"] = True
                events["play"] = False
                print("[yellow]Pressed ➡️. Forward to next episode[/yellow]")

        except Exception as e:
            print(f"[red]Error handling key press: {e}[/red]")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def play_episode(
    robot: RosRobot, policy: PolicyClient, fps: int, task: str, events: dict, logger: RerunLogger = None
):
    policy.reset()
    start_episode_t = time.perf_counter()
    dt = 1.0 / fps
    # Start control loop
    while True:
        # Start control loop
        start = time.perf_counter()

        # Request action from policy
        actions = policy.require_action().numpy()[:8]

        # Print actions for debugging
        with np.printoptions(precision=2, suppress=True, linewidth=800):
            print(actions)

        if logger:
            observation = policy.last_observation
            if observation is not None:
                data = {
                    "framestep": policy.timestep,
                    **observation,
                    "policy_action": actions,
                }
                logger.log(data)

        # Send action to the robot
        robot.send_action_real(actions)

        # Check if user requested to stop or forward
        if events.get("forward", False) or events.get("stop", False):
            break

        # Check if user requested to pause
        while not events["play"] and not events["stop"] and not events["forward"]:
            time.sleep(0.1)

        # Control loop timing to maintain the desired fps
        busy_wait(max(0.0, dt - (time.perf_counter() - start)))

    # Return the total time taken for the episode
    return time.perf_counter() - start_episode_t


@safe_disconnect
def play(robot: RosRobot, policy: PolicyClient, config: PlayConfig) -> LeRobotDataset:

    assert robot.is_connected, "Robot should be connected before recording."

    _, events = init_keyboard_listener()
    print("[yellow]Please use keyboard to recording...[/yellow]")

    logger = RerunLogger(url=config.rerun_url) if config.rerun_url else None

    while True:
        print("[yellow]Please wait for the robot to be ready[/yellow]")
        robot.reset()
        # Wait for the user to press the start button
        print(
            "[bright_yellow]Keyboard usage: [SPACE] to play/pause, [ESC] to stop, [→] to forward[/bright_yellow]"
        )
        # Reset events
        events["forward"] = False
        # Wait for user to start or stop, you should wait for robot to be ready
        while events and not events["play"] and not events["stop"]:
            time.sleep(0.1)

        # Check if stop requested
        if events["stop"]:
            print("Bye!")
            break

        if logger:
            logger.switch_record()

        # Start play an episode
        elapsed = play_episode(
            robot=robot, policy=policy, events=events, fps=config.fps, task=config.task, logger=logger
        )

        # Print episode summary
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print(f"[bright_yellow]Episode finished in {formatted_time}[/bright_yellow]")

        # Pause after each episode
        events["play"] = False

    if logger:
        logger.stop()

    print("[yellow]The show is over.[/yellow]")


def main(config: PlayConfig):
    print(asdict(config))

    robot = make_robot_from_config(config.robot)
    robot.connect()
    robot.set_task(config.task)

    policy = PolicyClient(config.policy, get_observation=robot.get_observation_tensor)

    play(robot, policy, config)

    robot.disconnect()


if __name__ == "__main__":
    main(tyro.cli(PlayConfig))
