import queue
import threading
import uuid

import numpy as np
import rerun as rr
import torch


class RerunLogger:
    """Non-blocking logger that streams robot data to local or remote Rerun viewer.

    Parameters
    ----------
    url: str
        Rerun endpoint URL. Determines where data is streamed:

        - ``rerun+http://127.0.0.1:9876/proxy`` — spawn & connect to a local viewer
        - ``rerun+http://localhost:9876/proxy``  — same as above (localhost alias)
        - ``rerun+http://192.168.1.10:9876/proxy`` — connect to a remote viewer over gRPC
    max_queue_size: int
        Maximum queued frames.  Oldest frame is dropped when full.
    """

    _CMD_LOG = 0
    _CMD_TIME_SEQ = 1
    _CMD_BLUEPRINT = 2
    _CMD_STOP = 3
    _CMD_NEW_RECORDING = 4

    def __init__(self, url: str, max_queue_size: int = 10, y_range: tuple[float, float] | None = None):
        self._url = url
        self._y_range = y_range  # None means auto-scale
        self._joint_count: int | None = None
        self._blueprint_sent = False
        self._next_frame_seq = 0
        # Camera slots are inferred from available observation.images.* keys.
        self._camera_slots: list[str] = []
        self._state_value_history: list[float] = []

        # Isolated recording stream - never touches global rr.init() state.
        self._rec = rr.RecordingStream(
            application_id="lerobot",
            recording_id=uuid.uuid4(),
            make_default=False,
            make_thread_default=False,
        )
        self._connect_stream()

        self._queue: queue.Queue[tuple] = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True, name="RerunLogger")
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _enqueue_blueprint(self) -> None:
        if self._blueprint_sent or self._joint_count is None:
            return

        camera_views = [
            rr.blueprint.Spatial2DView(
                origin="/",
                contents=[slot, f"overlays/{slot}/state_value_curve", f"overlays/{slot}/state_value_label"],
                name=slot.replace("_", " ").title(),
            )
            for slot in self._camera_slots
        ]
        joint_views = []
        for joint_no in range(1, self._joint_count + 1):
            joint_views.append(
                rr.blueprint.TimeSeriesView(
                    origin="/",
                    contents=[f"states/{joint_no}", f"teleop/{joint_no}", f"policy/{joint_no}"],
                    name=f"joint_{joint_no}",
                    axis_y=rr.blueprint.ScalarAxis(range=self._y_range) if self._y_range else None,
                )
            )

        top_row = None
        if camera_views:
            top_row = rr.blueprint.Grid(
                *camera_views,
                grid_columns=len(camera_views),
                row_shares=[1],
                column_shares=[1] * len(camera_views),
            )

        bottom_row = None
        if joint_views:
            bottom_row = rr.blueprint.Grid(
                *joint_views,
                grid_columns=min(4, len(joint_views)),
                row_shares=[1],
                column_shares=[1] * min(4, len(joint_views)),
            )

        if top_row and bottom_row:
            layout = rr.blueprint.Grid(
                top_row,
                bottom_row,
                grid_columns=1,
                row_shares=[1.5, 1],
                column_shares=[1],
            )
        elif top_row:
            layout = top_row
        elif bottom_row:
            layout = bottom_row
        else:
            return

        blueprint = rr.blueprint.Blueprint(layout)
        self._queue.put((self._CMD_BLUEPRINT, blueprint))
        self._blueprint_sent = True

    def log(self, data: dict) -> None:
        """Enqueue a data dict for async logging (non-blocking).

        Expected keys
        -------------
        ``observation.images.*``           : np.ndarray HWC uint8, dynamic camera count (1..3)
        ``observation.state``              : array-like, used to infer joint count
        ``teleop``                         : array-like (optional)
        ``policy``                         : array-like (optional)
        ``state_value``                    : scalar (optional), overlaid as a trend curve on camera views
        ``framestep``                      : int (optional). If missing, an internal increasing sequence is used.
        """
        if self._joint_count is None:
            image_keys = sorted(k for k in data if str(k).startswith("observation.images."))
            self._camera_slots = image_keys
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

    def _is_local(self) -> bool:
        return self._url and ("127.0.0.1" in self._url or "localhost" in self._url)

    def _connect_stream(self) -> None:
        if self._is_local():
            rr.spawn(recording=self._rec)
            return
        rr.connect_grpc(self._url, recording=self._rec)

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
                    self._connect_stream()
                    self._next_frame_seq = 0
                    self._state_value_history.clear()
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

    def _compute_state_value_points(self, image: np.ndarray) -> np.ndarray | None:
        if image.ndim < 2 or len(self._state_value_history) < 2:
            return None

        height, width = image.shape[0], image.shape[1]
        x0, x1 = width * 0.05, width * 0.95
        y0, y1 = height * 0.75, height * 0.9
        values = np.asarray(self._state_value_history, dtype=np.float32)
        v_min = float(values.min())
        v_max = float(values.max())
        if abs(v_max - v_min) < 1e-8:
            ys = np.full_like(values, (y0 + y1) * 0.5)
        else:
            ys = y1 - ((values - v_min) / (v_max - v_min)) * (y1 - y0)

        xs = np.linspace(x0, x1, num=len(values), dtype=np.float32)
        return np.stack([xs, ys.astype(np.float32)], axis=1)

    def _log_sync(self, data: dict) -> None:
        frame_seq = data.get("framestep")
        frame_seq = self._next_frame_seq if frame_seq is None else int(frame_seq)

        rr.set_time("frame", sequence=frame_seq, recording=self._rec)
        self._next_frame_seq = frame_seq + 1

        state_value = data.get("state_value")
        if state_value is not None:
            self._state_value_history.append(float(np.asarray(state_value).reshape(-1)[0]))

        for slot_name in self._camera_slots:
            if data.get(slot_name) is None:
                continue

            image = self._to_hwc_uint8_numpy(data[slot_name])
            rr.log(slot_name, rr.Image(image).compress(), recording=self._rec)

            points = self._compute_state_value_points(image)
            if points is not None:
                rr.log(
                    f"overlays/{slot_name}/state_value_curve",
                    rr.LineStrips2D([points], colors=[(32, 220, 128)], radii=[1.5]),
                    recording=self._rec,
                )
                latest_label_pos = points[-1] + np.array([8.0, -8.0], dtype=np.float32)
                rr.log(
                    f"overlays/{slot_name}/state_value_label",
                    rr.Points2D(
                        [latest_label_pos],
                        colors=[(32, 220, 128)],
                        labels=[f"{self._state_value_history[-1]:.3f}"],
                        show_labels=True,
                        radii=[0.0],
                    ),
                    recording=self._rec,
                )

        for i in range(self._joint_count):
            rr.log(f"states/{i + 1}", rr.Scalars(float(data["observation.state"][i])), recording=self._rec)
            if data.get("teleop") is not None:
                rr.log(f"teleop/{i + 1}", rr.Scalars(float(data["teleop"][i])), recording=self._rec)
            if data.get("policy") is not None:
                rr.log(f"policy/{i + 1}", rr.Scalars(float(data["policy"][i])), recording=self._rec)
