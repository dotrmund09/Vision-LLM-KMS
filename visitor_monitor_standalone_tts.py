#!/usr/bin/env python3
"""
Jetson Orin Nano - Visitor Monitor Standalone
==============================================

기능
----
1. CSI/USB 카메라 입력
2. YOLO-World TensorRT로 person + 의류 검출
3. ByteTrack으로 사람 추적
4. 각 사람에게 익명 visitor_id(P0001, P0002, ...) 부여
5. 상의/하의 ROI의 HSV 색상 분류
6. 최근 여러 프레임으로 인상착의 안정화
7. 하루 단위 JSON 방문 기록 저장
8. 지정 인상착의 감지:
      - 회색 계열 상의(gray/light_gray/dark_gray)
      - 검은색 하의(black)
9. 최근 12프레임 중 8프레임 이상 일치하면
      - "침입자 발생" TTS 음성 재생
      - JSON에 alert 기록
10. TTS는 Piper로 WAV를 한 번 생성한 뒤 aplay로 재생

기본 경로
---------
YOLO engine:
    src/models/YOLO/appearance_worldv2.engine

Piper:
    .piper_venv/bin/python
    src/models/Piper/ko_KR-kss-medium.onnx

방문 기록:
    src/output/visitor_logs/visitor_log_YYYY-MM-DD.json

경보 음성:
    src/audio/intruder_alert.wav
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. YOLO-World vocabulary
# ============================================================

WORLD_CLASSES = [
    "person",
    "t-shirt",
    "shirt",
    "long-sleeve shirt",
    "hoodie",
    "sweater",
    "jacket",
    "coat",
    "pants",
    "jeans",
    "shorts",
    "skirt",
    "dress",
    "hat",
    "cap",
    "backpack",
]

UPPER_TYPES = {
    "t-shirt",
    "shirt",
    "long-sleeve shirt",
    "hoodie",
    "sweater",
    "jacket",
    "coat",
    "dress",
}

LOWER_TYPES = {
    "pants",
    "jeans",
    "shorts",
    "skirt",
    "dress",
}


# ============================================================
# 2. 색상 정의
# ============================================================

COLOR_KO = {
    "black": "검정",
    "dark_gray": "진한 회색",
    "gray": "회색",
    "light_gray": "밝은 회색",
    "white": "흰색",
    "red": "빨강",
    "orange": "주황",
    "yellow": "노랑",
    "green": "초록",
    "cyan": "청록",
    "blue": "파랑",
    "purple": "보라",
    "pink": "분홍",
    "brown": "갈색",
    "beige": "베이지",
    "unknown": "미확인",
}

# 현재 지정 인상착의
TARGET_UPPER_COLORS = {
    "dark_gray",
    "gray",
    "light_gray",
}
TARGET_LOWER_COLORS = {
    "black",
}

ALERT_TEXT = "침입자 발생"


# ============================================================
# 3. Bounding box / ROI 함수
# ============================================================

def clamp_box(box, width, height):
    x1, y1, x2, y2 = [int(v) for v in box]

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    return x1, y1, x2, y2


def shrink_box(box, ratio=0.10):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    return (
        x1 + ratio * w,
        y1 + ratio * h,
        x2 - ratio * w,
        y2 - ratio * h,
    )


def body_part_box(person_box, part):
    """
    YOLO-World가 실제 의류 bbox를 검출하지 못한 경우
    person bbox에서 상의/하의 ROI를 근사한다.
    """
    x1, y1, x2, y2 = person_box

    w = x2 - x1
    h = y2 - y1

    if part == "upper":
        # 얼굴/손 영역을 최대한 제외
        return (
            x1 + 0.20 * w,
            y1 + 0.24 * h,
            x1 + 0.80 * w,
            y1 + 0.56 * h,
        )

    # lower
    return (
        x1 + 0.22 * w,
        y1 + 0.56 * h,
        x1 + 0.78 * w,
        y1 + 0.90 * h,
    )


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def center_inside(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    cx = 0.5 * (ix1 + ix2)
    cy = 0.5 * (iy1 + iy2)

    return (
        ox1 <= cx <= ox2
        and oy1 <= cy <= oy2
    )


def pick_garment(garments, person_box, valid_types):
    """
    하나의 person bbox에 속하는 의류 후보 중 가장 높은 confidence를 선택.
    """
    p_area = max(box_area(person_box), 1.0)
    candidates = []

    for garment in garments:

        if garment["name"] not in valid_types:
            continue

        if not center_inside(
            garment["box"],
            person_box,
        ):
            continue

        area_ratio = (
            box_area(garment["box"])
            / p_area
        )

        if area_ratio > 0.90:
            continue

        candidates.append(garment)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda g: g["conf"],
    )


# ============================================================
# 4. HSV 색상 판별
# ============================================================

def dominant_clothing_color(bgr_crop):
    """
    의류 ROI에서 dominant color를 HSV 기반으로 분류한다.

    주의:
    실제 환경의 조명/카메라에 따라 threshold calibration이 필요할 수 있다.
    """

    if (
        bgr_crop is None
        or bgr_crop.size == 0
    ):
        return "unknown", 0.0

    h_crop, w_crop = bgr_crop.shape[:2]

    if (
        h_crop < 8
        or w_crop < 8
    ):
        return "unknown", 0.0

    crop = cv2.GaussianBlur(
        bgr_crop,
        (5, 5),
        0,
    )

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    h, s, v = cv2.split(hsv)

    assigned = np.zeros(
        h.shape,
        dtype=bool,
    )

    counts = {}

    def add(name, mask):
        mask = (
            mask
            & (~assigned)
        )

        counts[name] = int(
            np.count_nonzero(mask)
        )

        assigned[mask] = True

    # --------------------------------------------------------
    # 무채색
    # --------------------------------------------------------

    add(
        "black",
        v < 58,
    )

    add(
        "white",
        (s < 42)
        & (v >= 218),
    )

    add(
        "light_gray",
        (s < 55)
        & (v >= 150)
        & (v < 218),
    )

    add(
        "gray",
        (s < 58)
        & (v >= 95)
        & (v < 150),
    )

    add(
        "dark_gray",
        (s < 60)
        & (v >= 58)
        & (v < 95),
    )

    # --------------------------------------------------------
    # 저채도 warm color
    # --------------------------------------------------------

    add(
        "beige",
        (h >= 8)
        & (h <= 28)
        & (s >= 18)
        & (s < 105)
        & (v >= 120),
    )

    add(
        "brown",
        (h >= 5)
        & (h <= 22)
        & (s >= 55)
        & (v >= 55)
        & (v < 170),
    )

    # --------------------------------------------------------
    # 유채색
    # OpenCV Hue = 0 ~ 179
    # --------------------------------------------------------

    chroma = (
        (s >= 55)
        & (v >= 55)
    )

    add(
        "red",
        chroma
        & (
            (h <= 8)
            | (h >= 172)
        ),
    )

    add(
        "orange",
        chroma
        & (h >= 9)
        & (h <= 22),
    )

    add(
        "yellow",
        chroma
        & (h >= 23)
        & (h <= 34),
    )

    add(
        "green",
        chroma
        & (h >= 35)
        & (h <= 85),
    )

    add(
        "cyan",
        chroma
        & (h >= 86)
        & (h <= 100),
    )

    add(
        "blue",
        chroma
        & (h >= 101)
        & (h <= 130),
    )

    add(
        "purple",
        chroma
        & (h >= 131)
        & (h <= 155),
    )

    add(
        "pink",
        chroma
        & (h >= 156)
        & (h <= 171),
    )

    total = int(h.size)

    if (
        total == 0
        or not counts
    ):
        return "unknown", 0.0

    name, count = max(
        counts.items(),
        key=lambda kv: kv[1],
    )

    ratio = count / total

    # 혼합색/배경 비율이 너무 높은 ROI는 unknown
    if ratio < 0.20:
        return "unknown", ratio

    return name, ratio


# ============================================================
# 5. 카메라
# ============================================================

def create_csi_pipeline(
    sensor_id=0,
    width=1280,
    height=720,
    fps=30,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), "
        f"width={width}, "
        f"height={height}, "
        f"framerate={fps}/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "queue leaky=downstream max-size-buffers=1 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def open_camera(args):

    if args.camera == "csi":

        pipeline = create_csi_pipeline(
            sensor_id=args.sensor_id,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )

        cap = cv2.VideoCapture(
            pipeline,
            cv2.CAP_GSTREAMER,
        )

    else:

        cap = cv2.VideoCapture(
            args.camera_index,
            cv2.CAP_V4L2,
        )

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            args.width,
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            args.height,
        )

        cap.set(
            cv2.CAP_PROP_FPS,
            args.fps,
        )

    if not cap.isOpened():
        raise RuntimeError(
            "Camera open failed. "
            "CSI/USB 카메라 연결 및 "
            "OpenCV GStreamer/V4L2 지원을 확인하세요."
        )

    return cap


def maybe_flip(frame, mode):

    if mode == "horizontal":
        return cv2.flip(frame, 1)

    if mode == "vertical":
        return cv2.flip(frame, 0)

    if mode == "both":
        return cv2.flip(frame, -1)

    return frame


# ============================================================
# 6. YOLO
# ============================================================

def load_model(
    model_path,
    use_world_prompts,
):
    model = YOLO(
        model_path
    )

    # base YOLO-World .pt를 직접 사용할 때만 필요
    if use_world_prompts:
        model.set_classes(
            WORLD_CLASSES
        )

    return model


def parse_result(result):

    detections = []

    if result.boxes is None:
        return detections

    names = result.names

    xyxy = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )

    conf = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    cls = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    if result.boxes.id is not None:

        ids = (
            result.boxes.id
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

    else:

        ids = np.full(
            len(xyxy),
            -1,
            dtype=int,
        )

    for (
        box,
        score,
        cls_id,
        track_id,
    ) in zip(
        xyxy,
        conf,
        cls,
        ids,
    ):

        detections.append(
            {
                "box": tuple(
                    float(x)
                    for x in box
                ),
                "conf": float(score),
                "cls_id": int(cls_id),
                "name": str(
                    names[int(cls_id)]
                ),
                "track_id": int(
                    track_id
                ),
            }
        )

    return detections


# ============================================================
# 7. 시간 / JSON
# ============================================================

def iso_now():
    return (
        datetime.now()
        .astimezone()
        .isoformat(
            timespec="seconds"
        )
    )


def today_str():
    return (
        datetime.now()
        .astimezone()
        .strftime("%Y-%m-%d")
    )


def hhmmss(timestamp):

    try:
        return (
            datetime
            .fromisoformat(timestamp)
            .strftime("%H:%M:%S")
        )

    except Exception:
        return timestamp


def seconds_between(
    start_time,
    end_time,
):

    try:

        start_dt = datetime.fromisoformat(
            start_time
        )

        end_dt = datetime.fromisoformat(
            end_time
        )

        return max(
            0,
            int(
                (
                    end_dt
                    - start_dt
                ).total_seconds()
            ),
        )

    except Exception:
        return 0


def atomic_write_json(
    path,
    data,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        temp_path,
        path,
    )


# ============================================================
# 8. 인상착의 smoothing
# ============================================================

def dominant_value(
    values,
    fallback="unknown",
):

    valid = [
        value
        for value in values
        if (
            value
            and value != "unknown"
        )
    ]

    if not valid:
        return fallback, 0.0

    value, count = (
        Counter(valid)
        .most_common(1)[0]
    )

    ratio = (
        count
        / len(valid)
    )

    return value, ratio


class AppearanceSmoother:

    def __init__(
        self,
        window=15,
    ):

        self.upper_color = deque(
            maxlen=window
        )

        self.upper_type = deque(
            maxlen=window
        )

        self.lower_color = deque(
            maxlen=window
        )

        self.lower_type = deque(
            maxlen=window
        )


    def update(
        self,
        observation,
    ):

        self.upper_color.append(
            observation[
                "upper"
            ][
                "color"
            ]
        )

        self.upper_type.append(
            observation[
                "upper"
            ][
                "type"
            ]
        )

        self.lower_color.append(
            observation[
                "lower"
            ][
                "color"
            ]
        )

        self.lower_type.append(
            observation[
                "lower"
            ][
                "type"
            ]
        )


    def stable(self):

        upper_color, upper_color_score = (
            dominant_value(
                self.upper_color
            )
        )

        upper_type, upper_type_score = (
            dominant_value(
                self.upper_type,
                "top",
            )
        )

        lower_color, lower_color_score = (
            dominant_value(
                self.lower_color
            )
        )

        lower_type, lower_type_score = (
            dominant_value(
                self.lower_type,
                "bottom",
            )
        )

        return {
            "upper": {
                "type": upper_type,
                "color": upper_color,
                "color_ko": COLOR_KO.get(
                    upper_color,
                    upper_color,
                ),
                "stability": round(
                    min(
                        upper_color_score,
                        upper_type_score,
                    ),
                    3,
                ),
            },

            "lower": {
                "type": lower_type,
                "color": lower_color,
                "color_ko": COLOR_KO.get(
                    lower_color,
                    lower_color,
                ),
                "stability": round(
                    min(
                        lower_color_score,
                        lower_type_score,
                    ),
                    3,
                ),
            },

            "description": (
                f"{COLOR_KO.get(upper_color, upper_color)} "
                f"{upper_type} / "
                f"{COLOR_KO.get(lower_color, lower_color)} "
                f"{lower_type}"
            ),
        }


# ============================================================
# 9. 하루 방문 기록
# ============================================================

class DailyVisitorLog:

    def __init__(
        self,
        output_dir,
    ):

        self.output_dir = Path(
            output_dir
        )

        self.date = today_str()

        self.path = (
            self.output_dir
            / f"visitor_log_{self.date}.json"
        )

        self.data = (
            self._load_or_create()
        )

        self.next_id = (
            self._next_id_number()
        )


    def _load_or_create(self):

        if self.path.exists():

            try:

                with open(
                    self.path,
                    "r",
                    encoding="utf-8",
                ) as f:

                    data = json.load(f)

                if "visitors" in data:
                    return data

            except Exception as e:

                print(
                    "[WARN] 기존 JSON 읽기 실패:",
                    e,
                )

        return {
            "date": self.date,
            "created_at": iso_now(),
            "updated_at": iso_now(),
            "target_appearance": {
                "upper_colors": sorted(
                    TARGET_UPPER_COLORS
                ),
                "lower_colors": sorted(
                    TARGET_LOWER_COLORS
                ),
                "tts_text": ALERT_TEXT,
            },
            "visitors": [],
        }


    def _next_id_number(self):

        highest = 0

        for visitor in self.data[
            "visitors"
        ]:

            visitor_id = str(
                visitor.get(
                    "visitor_id",
                    "",
                )
            )

            if (
                visitor_id.startswith("P")
                and visitor_id[1:].isdigit()
            ):

                highest = max(
                    highest,
                    int(
                        visitor_id[1:]
                    ),
                )

        return highest + 1


    def new_visitor_id(self):

        visitor_id = (
            f"P{self.next_id:04d}"
        )

        self.next_id += 1

        return visitor_id


    def get(
        self,
        visitor_id,
    ):

        for visitor in self.data[
            "visitors"
        ]:

            if (
                visitor[
                    "visitor_id"
                ]
                == visitor_id
            ):
                return visitor

        return None


    def create(
        self,
        visitor_id,
        tracker_id,
        timestamp,
        appearance,
    ):

        record = {
            "visitor_id": visitor_id,
            "tracker_id": int(
                tracker_id
            ),

            "first_seen": timestamp,
            "last_seen": timestamp,

            "first_seen_time": hhmmss(
                timestamp
            ),
            "last_seen_time": hhmmss(
                timestamp
            ),

            "duration_seconds": 0,

            "status": "present",

            "appearance": appearance,

            "appearance_history": [
                {
                    "timestamp": timestamp,
                    "description": appearance[
                        "description"
                    ],
                    "upper": appearance[
                        "upper"
                    ],
                    "lower": appearance[
                        "lower"
                    ],
                }
            ],

            "alerts": [],
        }

        self.data[
            "visitors"
        ].append(
            record
        )


    def update(
        self,
        visitor_id,
        timestamp,
        appearance,
    ):

        record = self.get(
            visitor_id
        )

        if record is None:
            return

        old_description = (
            record
            .get(
                "appearance",
                {},
            )
            .get(
                "description"
            )
        )

        record[
            "last_seen"
        ] = timestamp

        record[
            "last_seen_time"
        ] = hhmmss(
            timestamp
        )

        record[
            "duration_seconds"
        ] = seconds_between(
            record[
                "first_seen"
            ],
            timestamp,
        )

        record[
            "status"
        ] = "present"

        record[
            "appearance"
        ] = appearance

        if (
            appearance[
                "description"
            ]
            != old_description
        ):

            record[
                "appearance_history"
            ].append(
                {
                    "timestamp": timestamp,
                    "description": appearance[
                        "description"
                    ],
                    "upper": appearance[
                        "upper"
                    ],
                    "lower": appearance[
                        "lower"
                    ],
                }
            )


    def add_alert(
        self,
        visitor_id,
        timestamp,
        hits,
        window,
        appearance,
    ):

        record = self.get(
            visitor_id
        )

        if record is None:
            return

        record[
            "alerts"
        ].append(
            {
                "timestamp": timestamp,
                "time": hhmmss(
                    timestamp
                ),
                "event": "intruder_alert",
                "message": ALERT_TEXT,
                "match": {
                    "hits": int(hits),
                    "window": int(window),
                },
                "appearance": appearance,
            }
        )


    def save(self):

        self.data[
            "updated_at"
        ] = iso_now()

        atomic_write_json(
            self.path,
            self.data,
        )


# ============================================================
# 10. 사람별 인상착의 관측
# ============================================================

def observe_person(
    frame,
    person,
    garments,
):

    h_img, w_img = (
        frame.shape[:2]
    )

    person_box = clamp_box(
        person["box"],
        w_img,
        h_img,
    )

    upper_garment = pick_garment(
        garments,
        person_box,
        UPPER_TYPES,
    )

    lower_garment = pick_garment(
        garments,
        person_box,
        LOWER_TYPES,
    )

    if upper_garment:

        upper_box = shrink_box(
            upper_garment[
                "box"
            ],
            0.10,
        )

    else:

        upper_box = body_part_box(
            person_box,
            "upper",
        )

    if lower_garment:

        lower_box = shrink_box(
            lower_garment[
                "box"
            ],
            0.10,
        )

    else:

        lower_box = body_part_box(
            person_box,
            "lower",
        )

    ux1, uy1, ux2, uy2 = (
        clamp_box(
            upper_box,
            w_img,
            h_img,
        )
    )

    lx1, ly1, lx2, ly2 = (
        clamp_box(
            lower_box,
            w_img,
            h_img,
        )
    )

    upper_color, upper_ratio = (
        dominant_clothing_color(
            frame[
                uy1:uy2,
                ux1:ux2,
            ]
        )
    )

    lower_color, lower_ratio = (
        dominant_clothing_color(
            frame[
                ly1:ly2,
                lx1:lx2,
            ]
        )
    )

    return {
        "person_box": person_box,

        "upper_box": (
            ux1,
            uy1,
            ux2,
            uy2,
        ),

        "lower_box": (
            lx1,
            ly1,
            lx2,
            ly2,
        ),

        "upper": {
            "type": (
                upper_garment["name"]
                if upper_garment
                else "top"
            ),

            "color": upper_color,

            "color_ratio": round(
                upper_ratio,
                3,
            ),
        },

        "lower": {
            "type": (
                lower_garment["name"]
                if lower_garment
                else "bottom"
            ),

            "color": lower_color,

            "color_ratio": round(
                lower_ratio,
                3,
            ),
        },
    }


# ============================================================
# 11. 지정 인상착의 비교
# ============================================================

def is_target_appearance(
    stable_appearance,
):

    upper_color = (
        stable_appearance[
            "upper"
        ][
            "color"
        ]
    )

    lower_color = (
        stable_appearance[
            "lower"
        ][
            "color"
        ]
    )

    upper_match = (
        upper_color
        in TARGET_UPPER_COLORS
    )

    lower_match = (
        lower_color
        in TARGET_LOWER_COLORS
    )

    return (
        upper_match
        and lower_match
    )


# ============================================================
# 12. Piper TTS
# ============================================================

class IntruderTTS:
    """
    "침입자 발생" 음성을 한 번 생성해 두고,
    이후에는 aplay만 비동기로 실행한다.
    """

    def __init__(
        self,
        piper_python,
        piper_model,
        output_wav,
        speaker_device,
        enabled=True,
    ):

        self.piper_python = Path(
            piper_python
        )

        self.piper_model = Path(
            piper_model
        )

        self.output_wav = Path(
            output_wav
        )

        self.speaker_device = (
            speaker_device
        )

        self.enabled = enabled

        self.lock = threading.Lock()

        self.ready = False


    def prepare(self):

        if not self.enabled:
            print(
                "[TTS] 비활성화됨"
            )
            return False

        if not self.piper_python.exists():

            print(
                f"[TTS WARN] Piper Python 없음: "
                f"{self.piper_python}"
            )

            return False

        if not self.piper_model.exists():

            print(
                f"[TTS WARN] Piper model 없음: "
                f"{self.piper_model}"
            )

            return False

        self.output_wav.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 기존 경보 WAV가 있으면 그대로 재사용
        if self.output_wav.exists():

            self.ready = True

            print(
                f"[TTS] 기존 경보 음성 사용: "
                f"{self.output_wav}"
            )

            return True

        print(
            "[TTS] '침입자 발생' 음성 생성 중..."
        )

        try:

            subprocess.run(
                [
                    str(
                        self.piper_python
                    ),

                    "-m",
                    "piper",

                    "-m",
                    str(
                        self.piper_model
                    ),

                    "-f",
                    str(
                        self.output_wav
                    ),

                    "--",
                    ALERT_TEXT,
                ],
                check=True,
            )

            self.ready = (
                self.output_wav.exists()
            )

            if self.ready:

                print(
                    "[TTS] 경보 음성 생성 완료"
                )

            return self.ready

        except Exception as e:

            print(
                "[TTS ERROR] 음성 생성 실패:",
                e,
            )

            self.ready = False

            return False


    def _play_worker(self):

        # 이미 다른 경보 음성이 재생 중이면 중복 재생 안 함
        if not self.lock.acquire(
            blocking=False
        ):

            return

        try:

            subprocess.run(
                [
                    "aplay",
                    "-D",
                    self.speaker_device,
                    str(
                        self.output_wav
                    ),
                ],
                check=True,
            )

        except Exception as e:

            print(
                "[TTS ERROR] aplay 실패:",
                e,
            )

        finally:

            self.lock.release()


    def play(self):

        if (
            not self.enabled
            or not self.ready
        ):
            return

        thread = threading.Thread(
            target=self._play_worker,
            daemon=True,
        )

        thread.start()


# ============================================================
# 13. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        default=(
            "src/models/YOLO/"
            "appearance_worldv2.engine"
        ),
    )

    parser.add_argument(
        "--base-world",
        action="store_true",
    )


    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    parser.add_argument(
        "--camera",
        choices=[
            "csi",
            "usb",
        ],
        default="csi",
    )

    parser.add_argument(
        "--sensor-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--flip",
        choices=[
            "none",
            "horizontal",
            "vertical",
            "both",
        ],
        default="none",
    )


    # --------------------------------------------------------
    # Visitor log
    # --------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        default=(
            "src/output/"
            "visitor_logs"
        ),
    )

    parser.add_argument(
        "--appearance-window",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--leave-timeout",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--save-interval",
        type=float,
        default=1.0,
    )


    # --------------------------------------------------------
    # Intruder confirmation
    # --------------------------------------------------------

    parser.add_argument(
        "--alert-window",
        type=int,
        default=12,
        help=(
            "최근 몇 프레임을 "
            "경보 판단에 사용할지"
        ),
    )

    parser.add_argument(
        "--alert-hits",
        type=int,
        default=8,
        help=(
            "alert-window 중 "
            "몇 프레임 이상 일치해야 "
            "경보할지"
        ),
    )

    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=20.0,
        help=(
            "같은 visitor_id에 대해 "
            "TTS 재생 최소 간격(초)"
        ),
    )


    # --------------------------------------------------------
    # Piper TTS
    # --------------------------------------------------------

    parser.add_argument(
        "--piper-python",
        default=(
            ".piper_venv/bin/python"
        ),
    )

    parser.add_argument(
        "--piper-model",
        default=(
            "src/models/Piper/"
            "ko_KR-kss-medium.onnx"
        ),
    )

    parser.add_argument(
        "--alert-wav",
        default=(
            "src/audio/"
            "intruder_alert.wav"
        ),
    )

    parser.add_argument(
        "--speaker-device",
        default="plughw:2,0",
    )

    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="TTS 비활성화",
    )


    args = parser.parse_args()


    if (
        args.alert_hits
        > args.alert_window
    ):

        raise ValueError(
            "--alert-hits는 "
            "--alert-window 이하이어야 합니다."
        )


    # ========================================================
    # Model
    # ========================================================

    print(
        "[INFO] YOLO model loading..."
    )

    model = load_model(
        args.model,
        args.base_world,
    )

    print(
        "[INFO] YOLO model ready"
    )


    # ========================================================
    # Camera
    # ========================================================

    cap = open_camera(
        args
    )

    print(
        "[INFO] Camera ready"
    )


    # ========================================================
    # JSON log
    # ========================================================

    visitor_log = DailyVisitorLog(
        args.output_dir
    )

    print(
        "[INFO] JSON log:",
        visitor_log.path,
    )


    # ========================================================
    # TTS
    # ========================================================

    tts = IntruderTTS(
        piper_python=args.piper_python,
        piper_model=args.piper_model,
        output_wav=args.alert_wav,
        speaker_device=args.speaker_device,
        enabled=(
            not args.no_tts
        ),
    )

    tts.prepare()


    # ========================================================
    # Runtime state
    # ========================================================

    # ByteTrack ID -> 우리 visitor ID
    tracker_to_visitor = {}

    # ByteTrack ID -> appearance smoother
    appearance_smoothers = {}

    # ByteTrack ID -> 마지막으로 실제 보인 monotonic time
    last_seen_mono = {}

    # ByteTrack ID -> 최근 target appearance True/False
    alert_history = {}

    # visitor_id -> 마지막 경보 시간
    last_alert_time = {}

    last_json_save = 0.0


    print()
    print(
        "======================================"
    )
    print(
        " Target appearance"
    )
    print(
        " upper : gray / light_gray / dark_gray"
    )
    print(
        " lower : black"
    )
    print(
        f" confirm: "
        f"{args.alert_hits}/"
        f"{args.alert_window} frames"
    )
    print(
        f" TTS    : {ALERT_TEXT}"
    )
    print(
        " q      : quit"
    )
    print(
        "======================================"
    )
    print()


    try:

        while True:

            ok, frame = cap.read()

            if not ok:

                print(
                    "[ERROR] Camera frame read failed"
                )

                break


            frame = maybe_flip(
                frame,
                args.flip,
            )


            # =================================================
            # YOLO + ByteTrack
            # =================================================

            infer_kwargs = {
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "verbose": False,
            }

            # .pt인 경우만 PyTorch CUDA device 지정
            if not str(
                args.model
            ).endswith(
                ".engine"
            ):

                infer_kwargs[
                    "device"
                ] = 0


            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                **infer_kwargs,
            )[0]


            detections = parse_result(
                result
            )

            persons = [
                det
                for det in detections
                if det["name"] == "person"
            ]

            garments = [
                det
                for det in detections
                if det["name"] != "person"
            ]


            now_iso = iso_now()

            now_mono = time.monotonic()

            visible_tracker_ids = set()


            # =================================================
            # Person processing
            # =================================================

            for person in persons:

                tracker_id = int(
                    person[
                        "track_id"
                    ]
                )

                # ByteTrack ID가 아직 생성되지 않은 프레임
                if tracker_id < 0:
                    continue


                visible_tracker_ids.add(
                    tracker_id
                )

                last_seen_mono[
                    tracker_id
                ] = now_mono


                # ---------------------------------------------
                # 새로운 사람
                # ---------------------------------------------

                if (
                    tracker_id
                    not in tracker_to_visitor
                ):

                    visitor_id = (
                        visitor_log
                        .new_visitor_id()
                    )

                    tracker_to_visitor[
                        tracker_id
                    ] = visitor_id

                    appearance_smoothers[
                        tracker_id
                    ] = AppearanceSmoother(
                        args.appearance_window
                    )

                    alert_history[
                        tracker_id
                    ] = deque(
                        maxlen=args.alert_window
                    )


                visitor_id = (
                    tracker_to_visitor[
                        tracker_id
                    ]
                )


                # ---------------------------------------------
                # 인상착의 관측
                # ---------------------------------------------

                observation = observe_person(
                    frame,
                    person,
                    garments,
                )

                appearance_smoothers[
                    tracker_id
                ].update(
                    observation
                )

                stable_appearance = (
                    appearance_smoothers[
                        tracker_id
                    ].stable()
                )


                # ---------------------------------------------
                # JSON create/update
                # ---------------------------------------------

                if (
                    visitor_log.get(
                        visitor_id
                    )
                    is None
                ):

                    visitor_log.create(
                        visitor_id,
                        tracker_id,
                        now_iso,
                        stable_appearance,
                    )

                else:

                    visitor_log.update(
                        visitor_id,
                        now_iso,
                        stable_appearance,
                    )


                # =============================================
                # 지정 인상착의 판단
                # =============================================

                current_match = (
                    is_target_appearance(
                        stable_appearance
                    )
                )

                alert_history[
                    tracker_id
                ].append(
                    1
                    if current_match
                    else 0
                )

                hits = sum(
                    alert_history[
                        tracker_id
                    ]
                )

                samples = len(
                    alert_history[
                        tracker_id
                    ]
                )

                confirmed = (
                    samples
                    >= args.alert_window
                    and hits
                    >= args.alert_hits
                )


                # =============================================
                # 경보
                # =============================================

                if confirmed:

                    previous_alert = (
                        last_alert_time.get(
                            visitor_id,
                            -1e9,
                        )
                    )

                    if (
                        now_mono
                        - previous_alert
                        >= args.alert_cooldown
                    ):

                        last_alert_time[
                            visitor_id
                        ] = now_mono

                        print()
                        print(
                            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        )
                        print(
                            f"[INTRUDER] {visitor_id}"
                        )
                        print(
                            f"[INTRUDER] "
                            f"{stable_appearance['description']}"
                        )
                        print(
                            f"[INTRUDER] "
                            f"match={hits}/{samples}"
                        )
                        print(
                            f"[INTRUDER] {ALERT_TEXT}"
                        )
                        print(
                            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        )
                        print()

                        visitor_log.add_alert(
                            visitor_id,
                            now_iso,
                            hits,
                            samples,
                            stable_appearance,
                        )

                        # 즉시 JSON 저장
                        visitor_log.save()

                        # 비동기 음성 재생
                        tts.play()


                # =============================================
                # Visualization
                # =============================================

                (
                    px1,
                    py1,
                    px2,
                    py2,
                ) = observation[
                    "person_box"
                ]

                (
                    ux1,
                    uy1,
                    ux2,
                    uy2,
                ) = observation[
                    "upper_box"
                ]

                (
                    lx1,
                    ly1,
                    lx2,
                    ly2,
                ) = observation[
                    "lower_box"
                ]


                # confirmed면 빨간색, 아니면 초록색
                person_box_color = (
                    (0, 0, 255)
                    if confirmed
                    else (0, 255, 0)
                )


                cv2.rectangle(
                    frame,
                    (
                        px1,
                        py1,
                    ),
                    (
                        px2,
                        py2,
                    ),
                    person_box_color,
                    2,
                )


                # upper ROI
                cv2.rectangle(
                    frame,
                    (
                        ux1,
                        uy1,
                    ),
                    (
                        ux2,
                        uy2,
                    ),
                    (
                        255,
                        200,
                        0,
                    ),
                    1,
                )


                # lower ROI
                cv2.rectangle(
                    frame,
                    (
                        lx1,
                        ly1,
                    ),
                    (
                        lx2,
                        ly2,
                    ),
                    (
                        255,
                        0,
                        200,
                    ),
                    1,
                )


                label = (
                    f"{visitor_id} | "
                    f"{stable_appearance['upper']['color']} "
                    f"{stable_appearance['upper']['type']} / "
                    f"{stable_appearance['lower']['color']} "
                    f"{stable_appearance['lower']['type']} | "
                    f"match {hits}/{samples}"
                )


                cv2.putText(
                    frame,
                    label,
                    (
                        px1,
                        max(
                            25,
                            py1 - 8,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    person_box_color,
                    2,
                    cv2.LINE_AA,
                )


                if confirmed:

                    cv2.putText(
                        frame,
                        "INTRUDER",
                        (
                            px1,
                            min(
                                frame.shape[0] - 10,
                                py2 + 25,
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (
                            0,
                            0,
                            255,
                        ),
                        2,
                        cv2.LINE_AA,
                    )


            # =================================================
            # 퇴장 처리
            # =================================================

            stale_tracker_ids = []

            for (
                tracker_id,
                seen_time,
            ) in list(
                last_seen_mono.items()
            ):

                if (
                    tracker_id
                    in visible_tracker_ids
                ):
                    continue

                if (
                    now_mono
                    - seen_time
                    >= args.leave_timeout
                ):

                    visitor_id = (
                        tracker_to_visitor.get(
                            tracker_id
                        )
                    )

                    if visitor_id:

                        record = (
                            visitor_log.get(
                                visitor_id
                            )
                        )

                        if (
                            record
                            and record[
                                "status"
                            ]
                            == "present"
                        ):

                            record[
                                "status"
                            ] = "left"

                            record[
                                "closed_reason"
                            ] = "not_seen"


                    stale_tracker_ids.append(
                        tracker_id
                    )


            for tracker_id in (
                stale_tracker_ids
            ):

                last_seen_mono.pop(
                    tracker_id,
                    None,
                )

                tracker_to_visitor.pop(
                    tracker_id,
                    None,
                )

                appearance_smoothers.pop(
                    tracker_id,
                    None,
                )

                alert_history.pop(
                    tracker_id,
                    None,
                )


            # =================================================
            # JSON 주기 저장
            # =================================================

            if (
                now_mono
                - last_json_save
                >= args.save_interval
            ):

                visitor_log.save()

                last_json_save = (
                    now_mono
                )


            # =================================================
            # 화면 정보
            # =================================================

            cv2.putText(
                frame,
                (
                    f"Today visitors: "
                    f"{len(visitor_log.data['visitors'])}"
                ),
                (
                    20,
                    32,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (
                    0,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )


            cv2.putText(
                frame,
                (
                    "Target: GRAY upper + BLACK lower"
                ),
                (
                    20,
                    62,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (
                    0,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )


            cv2.imshow(
                "Visitor Appearance Logger + Intruder TTS",
                frame,
            )


            # q 종료
            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break


    finally:

        # 현재 보이는 사람들 종료 처리
        for visitor_id in (
            tracker_to_visitor.values()
        ):

            record = visitor_log.get(
                visitor_id
            )

            if (
                record
                and record[
                    "status"
                ]
                == "present"
            ):

                record[
                    "status"
                ] = "left"

                record[
                    "closed_reason"
                ] = "monitor_stopped"


        visitor_log.save()

        cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
