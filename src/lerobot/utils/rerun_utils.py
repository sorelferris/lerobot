#!/usr/bin/env python
import collections
import queue
import threading
import time
import uuid

import numpy as np
import rerun as rr
import torch

# Constants for overlay layout (avoid magic numbers)
_OVERLAY_X0, _OVERLAY_X1 = 0.05, 0.95
_OVERLAY_Y0, _OVERLAY_Y1 = 0.75, 0.9
_LABEL_OFFSET = np.array([8.0, -8.0], dtype=np.float32)
_OVERLAY_COLOR = (32, 220, 128)
_MAX_JOINT_COLS = 4


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
    _CMD_BLUEPRINT = 1
    _CMD_STOP = 2
    _CMD_NEW_RECORDING = 3

    def __init__(self, url: str, max_queue_size: int = 3):
        self._url = url
        self._joint_count: int | None = None
        self._blueprint_sent = False
        self._next_frame_seq = 0
        self._camera_slots: list[str] = []
        self._state_value_history: collections.deque[float] = collections.deque(maxlen=1000)
        self._lock = threading.Lock()

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
                grid_columns=min(_MAX_JOINT_COLS, len(joint_views)),
                row_shares=[1],
                column_shares=[1] * min(_MAX_JOINT_COLS, len(joint_views)),
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
            except queue.Empty:
                pass
            else:
                self._queue.task_done()
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
                    try:
                        self._log_sync(item[1])
                    except Exception:
                        # Log error but keep worker alive — rerun connection may recover
                        print(f"[RerunLogger] _log_sync error: {item[0][:80]}...", flush=True)
                elif cmd == self._CMD_BLUEPRINT:
                    rr.send_blueprint(item[1], recording=self._rec)
                elif cmd == self._CMD_NEW_RECORDING:
                    rr.disconnect(recording=self._rec)
                    self._rec = rr.RecordingStream(
                        application_id="lerobot",
                        recording_id=uuid.uuid4(),
                        make_default=False,
                        make_thread_default=False,
                    )
                    self._connect_stream()
                    self._next_frame_seq = 0
                    with self._lock:
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
        height, width = image.shape[0], image.shape[1]
        x0, x1 = width * _OVERLAY_X0, width * _OVERLAY_X1
        y0, y1 = height * _OVERLAY_Y0, height * _OVERLAY_Y1

        with self._lock:
            if len(self._state_value_history) < 2:
                return None
            values = np.asarray(self._state_value_history, dtype=np.float32)

        v_min, v_max = float(values.min()), float(values.max())
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
            with self._lock:
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
                    rr.LineStrips2D([points], colors=[_OVERLAY_COLOR], radii=[1.5]),
                    recording=self._rec,
                )
                with self._lock:
                    latest_value = self._state_value_history[-1]
                latest_label_pos = points[-1] + _LABEL_OFFSET
                rr.log(
                    f"overlays/{slot_name}/state_value_label",
                    rr.Points2D(
                        [latest_label_pos],
                        colors=[_OVERLAY_COLOR],
                        labels=[f"{latest_value:.3f}"],
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


class _StatsTracker:
    def __init__(self):
        self.start_time = time.time()
        self.frames_sent = 0
        self.frames_dropped = 0
        self.queue_sizes = []
        self.latencies = []

    def record_send(self, queue_size: int, latency: float):
        self.frames_sent += 1
        self.queue_sizes.append(queue_size)
        self.latencies.append(latency)

    def record_drop(self):
        self.frames_dropped += 1

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> dict:
        total = self.frames_sent + self.frames_dropped
        return {
            "elapsed_s": self.elapsed(),
            "frames_sent": self.frames_sent,
            "frames_dropped": self.frames_dropped,
            "drop_rate_pct": (self.frames_dropped / total * 100) if total > 0 else 0,
            "avg_fps": self.frames_sent / self.elapsed() if self.elapsed() > 0 else 0,
            "avg_queue_size": np.mean(self.queue_sizes) if self.queue_sizes else 0,
            "max_queue_size": max(self.queue_sizes) if self.queue_sizes else 0,
            "avg_latency_ms": (np.mean(self.latencies) * 1000) if self.latencies else 0,
        }


def _generate_mock_data(frame_seq: int, joint_count: int = 16) -> dict:
    image_vga = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image_720p = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    image_1080p = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    t = frame_seq * 0.01
    state = np.sin(t + np.arange(joint_count)) * 0.5
    teleop = state + np.random.randn(joint_count) * 0.05
    policy = state + np.random.randn(joint_count) * 0.02
    state_value = float(np.sin(t * 2) * 0.5 + 0.5)

    return {
        "observation.images.vga": image_vga,
        "observation.images.720p": image_720p,
        "observation.images.1080p": image_1080p,
        "observation.state": state,
        "teleop": teleop,
        "policy": policy,
        "state_value": state_value,
        "framestep": frame_seq,
    }


def _run_fps_phase(logger: RerunLogger, fps: int, duration_s: float, stats: _StatsTracker) -> None:
    interval = 1.0 / fps
    frame_seq = 0
    phase_start = time.time()
    last_report = time.time()

    while time.time() - phase_start < duration_s:
        send_time = time.time()
        data = _generate_mock_data(frame_seq)
        logger.log(data)

        queue_size = logger._queue.qsize()
        latency = time.time() - send_time
        stats.record_send(queue_size, latency)

        if time.time() - last_report >= 1.0:
            print(
                f"  {fps} FPS: sent={stats.frames_sent}, drop={stats.frames_dropped}, "
                f"qsize={queue_size}, latency={latency * 1000:.1f}ms"
            )
            last_report = time.time()

        frame_seq += 1
        elapsed_in_frame = time.time() - send_time
        sleep_time = max(0, interval - elapsed_in_frame)
        time.sleep(sleep_time)


def stress_test() -> None:
    print("=" * 60)
    print("RerunLogger Stress Test")
    print("=" * 60)

    with RerunLogger(url="rerun+http://172.20.76.73:9876/proxy", max_queue_size=20) as logger:
        print("\nPhase 1: Steady 30 FPS for 15 seconds")
        print("-" * 60)
        stats1 = _StatsTracker()
        _run_fps_phase(logger, 30, 15, stats1)

        print("\nPhase 2: Variable FPS (10→20→30→40→50) for 15 seconds")
        print("-" * 60)
        stats2 = _StatsTracker()
        fps_steps = [10, 20, 30, 40, 50]
        step_duration = 15 / len(fps_steps)
        for fps in fps_steps:
            print(f"\n  Switching to {fps} FPS...")
            _run_fps_phase(logger, fps, step_duration, stats2)

        print("\nFlushing queue...")
        logger.flush()

        print("\n" + "=" * 60)
        print("Final Results")
        print("=" * 60)

        s1 = stats1.summary()
        print(f"\nPhase 1 (30 FPS, 15s):")
        print(f"  Frames sent:     {s1['frames_sent']}")
        print(f"  Frames dropped:  {s1['frames_dropped']}")
        print(f"  Drop rate:       {s1['drop_rate_pct']:.2f}%")
        print(f"  Actual FPS:      {s1['avg_fps']:.1f}")
        print(f"  Avg queue size:  {s1['avg_queue_size']:.1f} (max={s1['max_queue_size']})")
        print(f"  Avg latency:     {s1['avg_latency_ms']:.1f}ms")

        s2 = stats2.summary()
        print(f"\nPhase 2 (Variable FPS, 15s):")
        print(f"  Frames sent:     {s2['frames_sent']}")
        print(f"  Frames dropped:  {s2['frames_dropped']}")
        print(f"  Drop rate:       {s2['drop_rate_pct']:.2f}%")
        print(f"  Actual FPS:      {s2['avg_fps']:.1f}")
        print(f"  Avg queue size:  {s2['avg_queue_size']:.1f} (max={s2['max_queue_size']})")
        print(f"  Avg latency:     {s2['avg_latency_ms']:.1f}ms")

        total_sent = s1["frames_sent"] + s2["frames_sent"]
        total_dropped = s1["frames_dropped"] + s2["frames_dropped"]
        total = total_sent + total_dropped
        print(f"\nOverall:")
        print(f"  Total frames:    {total}")
        print(f"  Total dropped:   {total_dropped} ({total_dropped / total * 100:.2f}%)")
        print("\nCheck Rerun viewer to verify visual display performance.")


if __name__ == "__main__":
    stress_test()
