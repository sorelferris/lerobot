import queue
import threading
import uuid

import numpy as np
import rerun as rr
import torch


class RerunLogger:
    """Non-blocking logger that streams robot data to a remote Rerun viewer over gRPC.

    Parameters
    ----------
    url:
        Remote Rerun gRPC proxy URL, e.g. ``rerun+http://172.20.76.88:9876/proxy``.
    max_queue_size:
        Maximum queued frames.  Oldest frame is dropped when full.
    """

    _CMD_LOG = 0
    _CMD_TIME_SEQ = 1
    _CMD_BLUEPRINT = 2
    _CMD_STOP = 3
    _CMD_NEW_RECORDING = 4

    def __init__(
        self,
        url: str | None = None,
        max_queue_size: int = 50,
    ):
        self._url = url if url else "rerun+http://127.0.0.1:9876/proxy"  # Default URL if none provided
        self._joint_count: int | None = None
        self._blueprint_sent = False

        # Isolated recording stream — never touches global rr.init() state.
        self._rec = rr.RecordingStream(
            application_id="lerobot",
            recording_id=uuid.uuid4(),
            make_default=False,
            make_thread_default=False,
        )
        rr.connect_grpc(url, recording=self._rec)

        self._queue: queue.Queue[tuple] = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True, name="RerunLogger")
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _enqueue_blueprint(self) -> None:
        if self._blueprint_sent or self._joint_count is None:
            return

        views = [
            rr.blueprint.Spatial2DView(origin="/", contents=["left_camera"], name="Left Camera"),
            rr.blueprint.Spatial2DView(origin="/", contents=["head_camera"], name="Head Camera"),
            rr.blueprint.Spatial2DView(origin="/", contents=["right_camera"], name="Right Camera"),
            rr.blueprint.Spatial2DView(origin="/", contents=["blank_camera"], name="Blank Camera"),
        ]
        for i in range(self._joint_count):
            views.append(
                rr.blueprint.TimeSeriesView(
                    origin="/",
                    contents=[f"states/{i + 1}", f"teleop/{i + 1}", f"policy/{i + 1}"],
                    name=f"joint_{i + 1}",
                )
            )
        grid_columns = 4
        total_rows = (len(views) + grid_columns - 1) // grid_columns
        row_shares = [2] + [1] * (total_rows - 1)
        blueprint = rr.blueprint.Blueprint(
            rr.blueprint.Grid(
                *views,
                grid_columns=grid_columns,
                row_shares=row_shares,
                column_shares=[1, 1, 1, 1],
            )
        )
        self._queue.put((self._CMD_BLUEPRINT, blueprint))
        self._blueprint_sent = True

    def log(self, data: dict) -> None:
        """Enqueue a data dict for async logging (non-blocking).

        Expected keys
        -------------
        ``observation.images.head_camera`` : np.ndarray  HWC uint8
        ``observation.images.*``           : np.ndarray  (any other camera)
        ``observation.state``              : array-like, used to infer joint count
        ``framestep``                      : int
        ``teleop_action``                  : array-like (optional)
        ``policy_action``                  : array-like (optional)
        """
        if self._joint_count is None:
            self._joint_count = len(data["observation.state"])
            self._enqueue_blueprint()

        try:
            self._queue.put_nowait((self._CMD_LOG, data))
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait((self._CMD_LOG, data))

    def switch_record(self) -> None:
        """Switch to a new recording_id (e.g., when starting a new episode).

        This creates a new RecordingStream instance in the worker thread,
        allowing each episode to appear as a separate recording in the viewer.
        Also resets the blueprint so it is re-sent for the new recording.
        """
        self._blueprint_sent = False
        self._queue.put((self._CMD_NEW_RECORDING,))

    def flush(self) -> None:
        """Block until all queued items have been sent to the network."""
        self._queue.join()

    def stop(self) -> None:
        """Flush, stop the worker thread, and disconnect."""
        self.flush()
        self._queue.put((self._CMD_STOP,))
        self._thread.join(timeout=3.0)
        rr.disconnect(recording=self._rec)

    def __enter__(self) -> "RerunLogger":
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                cmd = item[0]
                if cmd == self._CMD_STOP:
                    return
                elif cmd == self._CMD_LOG:
                    self._log_sync(item[1])
                elif cmd == self._CMD_BLUEPRINT:
                    rr.send_blueprint(item[1], recording=self._rec)
                elif cmd == self._CMD_NEW_RECORDING:
                    # Disconnect old recording and create a new one with fresh recording_id
                    rr.disconnect(recording=self._rec)
                    self._rec = rr.RecordingStream(
                        application_id="lerobot",
                        recording_id=uuid.uuid4(),
                        make_default=False,
                        make_thread_default=False,
                    )
                    rr.connect_grpc(self._url, recording=self._rec)
            finally:
                self._queue.task_done()

    def _to_hwc_uint8_numpy(self, image: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] <= 4:
                if image.dtype == torch.uint8:
                    return image.permute(1, 2, 0).cpu().numpy()
                return (image.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            return image.cpu().numpy()

        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[-1] > 4:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                scale = 255.0 if arr.max(initial=0.0) <= 1.0 else 1.0
                arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _log_sync(self, data: dict) -> None:
        rr.set_time("frame", sequence=int(data["framestep"]), recording=self._rec)

        head_image = self._to_hwc_uint8_numpy(data["observation.images.head_camera"])
        left_raw = data.get("observation.images.left_camera")
        right_raw = data.get("observation.images.right_camera")
        left_image = self._to_hwc_uint8_numpy(left_raw) if left_raw is not None else np.zeros_like(head_image)
        right_image = (
            self._to_hwc_uint8_numpy(right_raw) if right_raw is not None else np.zeros_like(head_image)
        )
        blank_image = np.zeros_like(head_image)
        rr.log("head_camera", rr.Image(head_image), recording=self._rec)
        rr.log("left_camera", rr.Image(left_image), recording=self._rec)
        rr.log("right_camera", rr.Image(right_image), recording=self._rec)
        rr.log("blank_camera", rr.Image(blank_image), recording=self._rec)

        for i in range(self._joint_count):
            rr.log(f"states/{i + 1}", rr.Scalars(float(data["observation.state"][i])), recording=self._rec)
            if data.get("teleop_action") is not None:
                rr.log(f"teleop/{i + 1}", rr.Scalars(float(data["teleop_action"][i])), recording=self._rec)
            if data.get("policy_action") is not None:
                rr.log(f"policy/{i + 1}", rr.Scalars(float(data["policy_action"][i])), recording=self._rec)


if __name__ == "__main__":
    import argparse
    import time

    from lerobot.utils.replay_bot import ReplayRobot, ReplayRobotConfig

    def pick_camera(observation: dict, token: str):
        image_keys = [k for k in observation if "observation.images" in k]
        for key in image_keys:
            if token in key.lower():
                return observation[key]
        return None

    parser = argparse.ArgumentParser(description="Replay dataset episode(s) and stream them to remote Rerun")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo id, e.g. lerobot/pusht")
    parser.add_argument("--root", default=None, help="Optional local dataset root")
    parser.add_argument("--episode", default="0", help="Episode index or comma-separated indices")
    parser.add_argument("--url", default="rerun+http://172.20.76.88:9876/proxy", help="Remote rerun gRPC URL")
    parser.add_argument("--dt", type=float, default=0.02, help="Sleep seconds between frames")
    args = parser.parse_args()

    robot_cfg = ReplayRobotConfig(repo_id=args.repo_id, root=args.root, episode=args.episode)
    robot = ReplayRobot(robot_cfg)

    sent_frames = 0
    with RerunLogger(url=args.url) as logger:
        for episode_index in robot.episodes:
            robot.load_episode(episode_index)
            logger.switch_record()  # Switch to new recording for this episode
            while not robot.is_episode_done:
                observation = robot.get_observation()
                if observation is None:
                    break

                head = pick_camera(observation, "head")
                if head is None:
                    raise KeyError("Replay observation does not contain a head camera image")

                left = pick_camera(observation, "left")
                right = pick_camera(observation, "right")

                data = {
                    "observation.images.head_camera": head,
                    "observation.state": observation["observation.state"],
                    "teleop_action": robot.teleop_action,
                    "framestep": sent_frames,
                }
                if left is not None:
                    data["observation.images.left_camera"] = left
                if right is not None:
                    data["observation.images.right_camera"] = right

                logger.log(data)
                robot.send_action({})
                sent_frames += 1
                time.sleep(args.dt)

    print(f"Done — sent {sent_frames} frames to {args.url}")
