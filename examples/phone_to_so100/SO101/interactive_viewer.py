#!/usr/bin/env python3
"""Interactive MuJoCo viewer for the SO101 robot arm.

Provides a reusable ``SO101Viewer`` component that can be driven externally::

    from interactive_viewer import SO101Viewer

    view = SO101Viewer()
    # actions in actuator order: [shoulder_pan, shoulder_lift, elbow_flex,
    #                              wrist_flex, wrist_roll, gripper]  (radians)
    view.send_action([0.0, -0.5, 0.8, 0.0, 0.0, 0.0])

    # or by name:
    view.send_action({"shoulder_pan": 0.3, "gripper": 1.2})

    view.close()

When run directly it launches a standalone viewer with UI sliders::

    python interactive_viewer.py
"""

import pathlib
from collections.abc import Sequence

import mujoco
import mujoco.viewer

MODEL_DIR = pathlib.Path(__file__).parent
SCENE_XML = MODEL_DIR / "scene.xml"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class SO101Viewer:
    """MuJoCo viewer for the SO101 arm with ``send_action`` API.

    Parameters
    ----------
    scene_path : str or Path, optional
        Path to the scene MJCF XML.  Defaults to the bundled scene.xml.
    kp : float, optional
        Override actuator proportional gain (default keeps XML value).
    kv : float, optional
        Override actuator derivative gain (default keeps XML value).
    """

    def __init__(
        self,
        scene_path: str | pathlib.Path | None = None,
        kp: float | None = None,
        kv: float | None = None,
    ):
        self._m = mujoco.MjModel.from_xml_path(str(scene_path or SCENE_XML))
        self._d = mujoco.MjData(self._m)

        # Actuator name -> ctrl index lookup
        self._name_to_idx: dict[str, int] = {}
        for i in range(self._m.nu):
            name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self._name_to_idx[name] = i

        # Optional gain override
        if kp is not None or kv is not None:
            for i in range(self._m.nu):
                if kp is not None:
                    self._m.actuator_gainprm[i, 0] = kp
                if kv is not None:
                    self._m.actuator_biasprm[i, 2] = -kv

        mujoco.mj_forward(self._m, self._d)

        # Launch viewer (non-blocking background thread)
        self._viewer = mujoco.viewer.launch_passive(
            self._m, self._d, show_right_ui=True
        )

    def send_action(self, actions: dict[str, float] | Sequence[float]) -> None:
        """Set actuator targets.

        Parameters
        ----------
        actions : dict or sequence of float
            If a dict, keys are joint names and values are angles in radians.
            If a list/array, must be 6 values in actuator order:
            [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
             wrist_roll, gripper].
        """
        if isinstance(actions, dict):
            for name, val in actions.items():
                idx = self._name_to_idx.get(name)
                if idx is not None:
                    self._d.ctrl[idx] = val
        else:
            for i, val in enumerate(actions):
                if i < self._m.nu:
                    self._d.ctrl[i] = val

    @property
    def model(self) -> mujoco.MjModel:
        return self._m

    @property
    def data(self) -> mujoco.MjData:
        return self._d

    @property
    def joint_names(self) -> list[str]:
        return list(self._name_to_idx.keys())

    @property
    def is_running(self) -> bool:
        return self._viewer.is_running()

    def step(self) -> None:
        """Advance the simulation by one timestep and sync the viewer."""
        mujoco.mj_step(self._m, self._d)
        self._viewer.sync()

    def close(self) -> None:
        self._viewer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


# ── standalone demo ──────────────────────────────────────────────────────────


def main():
    view = SO101Viewer()

    print("=" * 55)
    print("  SO101 Robot Arm - Interactive MuJoCo Viewer")
    print("=" * 55)
    print(f"  Joints: {view.joint_names}")
    print("  Drag sliders on the right panel to move joints.")
    print("  Mouse: rotate (left) / pan (right) / zoom (scroll)")
    print("  Space: pause | Backspace: reset | Esc: quit")
    print("=" * 55)

    while view.is_running:
        view.step()


if __name__ == "__main__":
    main()
