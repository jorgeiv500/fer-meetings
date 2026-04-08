def sample_clip_windows(duration_s, clip_seconds, stride_seconds, start_offset_seconds=0.0, max_clips=None):
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be > 0")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be > 0")

    current = max(0.0, float(start_offset_seconds))
    windows = []
    while current + clip_seconds <= duration_s + 1e-9:
        windows.append((round(current, 3), round(current + clip_seconds, 3)))
        current += stride_seconds
        if max_clips is not None and len(windows) >= max_clips:
            break
    return windows


def sample_frame_times(start_s, end_s, frames_per_clip):
    if frames_per_clip < 1:
        raise ValueError("frames_per_clip must be >= 1")
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s")

    if frames_per_clip == 1:
        return [round((start_s + end_s) / 2.0, 3)]

    span = end_s - start_s
    step = span / (frames_per_clip + 1)
    return [round(start_s + (index + 1) * step, 3) for index in range(frames_per_clip)]


def majority_vote(labels, tie_order):
    if not labels:
        return ""

    counts = {label: 0 for label in tie_order}
    for label in labels:
        if label in counts:
            counts[label] += 1

    return max(tie_order, key=lambda label: counts[label])
