import argparse
import select
import sys
import tempfile
import termios
import time
import tty
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import numpy as np
import samna
import sinabs.backend.dynapcnn.io as sio


DEVICE = "speck2fdevkit:0"
SENSOR_WIDTH = 128
SENSOR_HEIGHT = 128
TIMESTAMP_MODULUS = 1 << 32
TIMESTAMP_WRAP_THRESHOLD = 1 << 31
EVENT_DTYPE = np.dtype([("x", "u1"), ("y", "u1"), ("p", bool), ("t", "u4")])


def import_opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCVがありません。更新したspeck.defからSIFを再ビルドしてください。"
        ) from exc
    return cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record Speck DVS events until a key is pressed and save an MP4."
    )
    parser.add_argument("--device", default=DEVICE, help="Sinabs device identifier")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/dvs_recording.mp4"),
        help="output video path (default: output/dvs_recording.mp4)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="output frame rate")
    parser.add_argument(
        "--gain",
        type=float,
        default=32.0,
        help="brightness added by one event (default: 32)",
    )
    parser.add_argument(
        "--stop-key",
        default="q",
        help="single key used to stop recording (default: q)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.01,
        help="seconds between event-buffer reads (default: 0.01)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop automatically after this many seconds instead of waiting for a key",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="four-character OpenCV codec (default: mp4v)",
    )
    args = parser.parse_args()

    if len(args.stop_key) != 1:
        parser.error("--stop-key must be exactly one character")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.gain <= 0:
        parser.error("--gain must be positive")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if len(args.codec) != 4:
        parser.error("--codec must contain exactly four characters")

    return args


def dvs_events_to_numpy(events: List[samna.speck2f.event.DvsEvent]) -> np.ndarray:
    return np.array(
        [
            (ev.x, ev.y, ev.p, ev.timestamp)
            for ev in events
            if isinstance(ev, samna.speck2f.event.DvsEvent)
        ],
        dtype=EVENT_DTYPE,
    )


class TerminalKeyReader:
    """Read one key from an interactive POSIX terminal without waiting for Enter."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._original_settings: Optional[list] = None

    def __enter__(self) -> "TerminalKeyReader":
        if not sys.stdin.isatty():
            raise RuntimeError(
                "standard input is not an interactive terminal; "
                "run from an Apptainer shell or use --duration"
            )
        self._original_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._original_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_settings)

    def read(self) -> Optional[str]:
        if select.select([sys.stdin], [], [], 0.0)[0]:
            return sys.stdin.read(1)
        return None


class TimestampUnwrapper:
    """Convert monotonically increasing uint32 timestamps to uint64 timestamps."""

    def __init__(self) -> None:
        self._last_raw: Optional[int] = None
        self._wrap_count = 0

    def unwrap(self, raw_timestamps: np.ndarray) -> np.ndarray:
        if len(raw_timestamps) == 0:
            return np.empty(0, dtype=np.uint64)

        signed = raw_timestamps.astype(np.int64)
        previous = np.empty_like(signed)
        previous[0] = signed[0] if self._last_raw is None else self._last_raw
        previous[1:] = signed[:-1]

        wraps = (signed - previous) < -TIMESTAMP_WRAP_THRESHOLD
        cumulative_wraps = np.cumsum(wraps, dtype=np.uint64)
        unwrapped = raw_timestamps.astype(np.uint64) + (
            self._wrap_count + cumulative_wraps
        ) * TIMESTAMP_MODULUS

        self._wrap_count += int(cumulative_wraps[-1])
        self._last_raw = int(raw_timestamps[-1])
        return unwrapped


def record_events(
    sink,
    stopwatch,
    raw_path: Path,
    stop_key: str,
    poll_interval: float,
    duration: Optional[float],
) -> tuple[int, float]:
    event_count = 0
    stopwatch.start()
    stopwatch.reset()
    sink.clear_events()
    started_at = time.monotonic()

    key_context = TerminalKeyReader() if duration is None else nullcontext(None)
    if duration is None:
        print(f"DVS記録中です。停止するには {stop_key!r} を押してください。")
    else:
        print(f"DVSイベントを{duration:g}秒間記録します。")

    with raw_path.open("wb") as raw_file:
        try:
            with key_context as key_reader:
                while True:
                    chunk = dvs_events_to_numpy(sink.get_events())
                    if len(chunk) > 0:
                        chunk.tofile(raw_file)
                        event_count += len(chunk)

                    elapsed = time.monotonic() - started_at
                    if duration is not None and elapsed >= duration:
                        break

                    if key_reader is not None:
                        key = key_reader.read()
                        if key is not None and key.lower() == stop_key.lower():
                            break

                    time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\nCtrl-Cを受け取り、記録を停止します。")
        finally:
            # Include events that arrived between the final poll and the key press.
            chunk = dvs_events_to_numpy(sink.get_events())
            if len(chunk) > 0:
                chunk.tofile(raw_file)
                event_count += len(chunk)

    return event_count, time.monotonic() - started_at


def render_frame(
    positive_counts: np.ndarray, negative_counts: np.ndarray, gain: float
) -> np.ndarray:
    # OpenCV uses BGR. Match Samna Visualizer: p=0 is green and p=1 is red.
    frame = np.zeros((SENSOR_HEIGHT, SENSOR_WIDTH, 3), dtype=np.uint8)
    frame[:, :, 1] = np.clip(negative_counts * gain, 0, 255).astype(np.uint8)
    frame[:, :, 2] = np.clip(positive_counts * gain, 0, 255).astype(np.uint8)
    return frame


def add_events_to_frame(
    events: np.ndarray,
    positive_counts: np.ndarray,
    negative_counts: np.ndarray,
) -> None:
    valid = (events["x"] < SENSOR_WIDTH) & (events["y"] < SENSOR_HEIGHT)
    events = events[valid]
    positive = events["p"]

    np.add.at(
        positive_counts,
        (events["y"][positive], events["x"][positive]),
        1,
    )
    np.add.at(
        negative_counts,
        (events["y"][~positive], events["x"][~positive]),
        1,
    )


def convert_events_to_video(
    raw_path: Path,
    event_count: int,
    recording_duration: float,
    output_path: Path,
    fps: float,
    gain: float,
    codec: str,
) -> int:
    cv2 = import_opencv()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (SENSOR_WIDTH, SENSOR_HEIGHT),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"動画ファイルを開けませんでした: {output_path} (codec={codec})"
        )

    frame_period_us = max(1, round(1_000_000 / fps))
    duration_frames = max(1, int(np.ceil(recording_duration * fps)))
    current_frame_index = 0
    positive_counts = np.zeros((SENSOR_HEIGHT, SENSOR_WIDTH), dtype=np.uint32)
    negative_counts = np.zeros((SENSOR_HEIGHT, SENSOR_WIDTH), dtype=np.uint32)
    unwrapper = TimestampUnwrapper()

    def write_current_frame() -> None:
        nonlocal current_frame_index
        writer.write(render_frame(positive_counts, negative_counts, gain))
        positive_counts.fill(0)
        negative_counts.fill(0)
        current_frame_index += 1

    try:
        if event_count > 0:
            recorded_events = np.memmap(
                raw_path,
                dtype=EVENT_DTYPE,
                mode="r",
                shape=(event_count,),
            )
            chunk_size = 1_000_000

            for chunk_start in range(0, event_count, chunk_size):
                chunk = recorded_events[chunk_start : chunk_start + chunk_size]
                timestamps = unwrapper.unwrap(chunk["t"])
                frame_indices = timestamps // frame_period_us
                boundaries = np.flatnonzero(np.diff(frame_indices)) + 1
                starts = np.concatenate(([0], boundaries))
                ends = np.concatenate((boundaries, [len(chunk)]))

                for start, end in zip(starts, ends):
                    target_frame_index = int(frame_indices[start])
                    if target_frame_index < current_frame_index:
                        target_frame_index = current_frame_index

                    while current_frame_index < target_frame_index:
                        write_current_frame()

                    add_events_to_frame(
                        chunk[start:end], positive_counts, negative_counts
                    )

            del recorded_events
            # A device timestamp can lead the host timer slightly. Always flush
            # the frame that received the last event in that case.
            duration_frames = max(duration_frames, current_frame_index + 1)

        while current_frame_index < duration_frames:
            write_current_frame()
    finally:
        writer.release()

    return current_frame_index


def main() -> None:
    args = parse_args()
    # Fail before opening Speck or recording anything if video support is absent.
    try:
        import_opencv()
    except RuntimeError as exc:
        raise SystemExit(f"エラー: {exc}") from None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{args.output.stem}.",
        suffix=".events",
        dir=args.output.parent,
        delete=False,
    )
    raw_path = Path(temporary_file.name)
    temporary_file.close()

    device_open = False
    video_complete = False
    try:
        devkit = sio.open_device(args.device)
        device_open = True
        sink = samna.graph.sink_from(devkit.get_model_source_node())

        config = samna.speck2f.configuration.SpeckConfiguration()
        config.dvs_layer.raw_monitor_enable = True
        devkit.get_model().apply_configuration(config)

        event_count, recording_duration = record_events(
            sink=sink,
            stopwatch=devkit.get_stop_watch(),
            raw_path=raw_path,
            stop_key=args.stop_key,
            poll_interval=args.poll_interval,
            duration=args.duration,
        )
        print(
            f"記録終了: {recording_duration:.2f}秒、"
            f"{event_count:,}イベント。動画へ変換します。"
        )

        sio.close_device(args.device)
        device_open = False

        frame_count = convert_events_to_video(
            raw_path=raw_path,
            event_count=event_count,
            recording_duration=recording_duration,
            output_path=args.output,
            fps=args.fps,
            gain=args.gain,
            codec=args.codec,
        )
        video_complete = True
        print(f"動画を保存しました: {args.output} ({frame_count}フレーム)")
    finally:
        if device_open:
            try:
                sio.close_device(args.device)
            except Exception as exc:
                print(f"警告: デバイスを閉じられませんでした: {exc}", file=sys.stderr)
        if video_complete or raw_path.stat().st_size == 0:
            raw_path.unlink(missing_ok=True)
        else:
            print(
                f"動画変換に失敗したため、一時イベントを保存しました: {raw_path}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
