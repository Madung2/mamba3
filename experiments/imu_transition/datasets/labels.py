"""Label metadata for the UCI HAPT dataset."""

from __future__ import annotations

ACTIVITY_LABELS = {
    1: "walking",
    2: "walking_upstairs",
    3: "walking_downstairs",
    4: "sitting",
    5: "standing",
    6: "lying",
    7: "stand_to_sit",
    8: "sit_to_stand",
    9: "sit_to_lie",
    10: "lie_to_sit",
    11: "stand_to_lie",
    12: "lie_to_stand",
}

TRANSITION_IDS = {7, 8, 9, 10, 11, 12}

# 7-class direction task: 0 = non-transition (any base activity),
# 1..6 = directed transitions in the order defined here.
DIRECTION_MAP = {
    7: 1,   # stand_to_sit
    8: 2,   # sit_to_stand
    9: 3,   # sit_to_lie
    10: 4,  # lie_to_sit
    11: 5,  # stand_to_lie
    12: 6,  # lie_to_stand
}
DIRECTION_CLASS_NAMES = [
    "non_transition",
    "stand_to_sit",
    "sit_to_stand",
    "sit_to_lie",
    "lie_to_sit",
    "stand_to_lie",
    "lie_to_stand",
]
NUM_DIRECTION_CLASSES = len(DIRECTION_CLASS_NAMES)


def to_direction_label(activity_id: int) -> int:
    """0 for any non-transition activity, otherwise 1..6 per DIRECTION_MAP."""
    return int(DIRECTION_MAP.get(int(activity_id), 0))

CHANNEL_MODES = {
    "acc": (0, 1, 2),
    "gyro": (3, 4, 5),
    "acc_gyro": (0, 1, 2, 3, 4, 5),
}


def is_transition(activity_id: int) -> int:
    """Return 1 for transition windows and 0 otherwise."""

    return int(activity_id in TRANSITION_IDS)


def resolve_channel_indices(channel_mode: str) -> tuple[int, ...]:
    """Map a channel mode string to feature indices."""

    if channel_mode not in CHANNEL_MODES:
        raise ValueError(
            f"Unsupported channel mode '{channel_mode}'. Expected one of {sorted(CHANNEL_MODES)}."
        )
    return CHANNEL_MODES[channel_mode]
