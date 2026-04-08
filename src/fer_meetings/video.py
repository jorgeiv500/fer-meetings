from pathlib import Path


def resolve_video_path(raw_dir, requested_name):
    requested = Path(requested_name)
    if requested.is_file():
        return requested.resolve()

    root = Path(raw_dir)
    direct_candidate = root / requested_name
    if direct_candidate.is_file():
        return direct_candidate.resolve()

    matches = list(root.rglob(requested.name))
    if not matches:
        raise FileNotFoundError(f"Could not find video '{requested_name}' inside '{raw_dir}'")
    if len(matches) > 1:
        raise FileExistsError(f"Multiple matches found for '{requested_name}': {matches}")
    return matches[0].resolve()


def parse_video_name(video_path):
    parts = Path(video_path).name.split(".")
    meeting_id = parts[0]
    camera = parts[1] if len(parts) >= 3 else "Unknown"
    return meeting_id, camera


def get_video_metadata(video_path):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = frame_count / fps if fps > 0 else 0.0
    capture.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": round(duration_s, 3),
    }


def open_video(video_path):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return capture


def read_frame_at(capture, fps, timestamp_s):
    import cv2

    target_frame = max(0, int(round(timestamp_s * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = capture.read()
    if not ok:
        return None
    return frame
