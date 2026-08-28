#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


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


def clamp_box(box, width, height):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def shrink_box(box, ratio=0.12):
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
    x1, y1, x2, y2 = person_box
    w = x2 - x1
    h = y2 - y1

    if part == "upper":
        return (
            x1 + 0.20 * w,
            y1 + 0.24 * h,
            x1 + 0.80 * w,
            y1 + 0.56 * h,
        )

    return (
        x1 + 0.22 * w,
        y1 + 0.56 * h,
        x1 + 0.78 * w,
        y1 + 0.90 * h,
    )


def dominant_clothing_color(bgr_crop):
    if bgr_crop is None or bgr_crop.size == 0:
        return "unknown", 0.0

    hh, ww = bgr_crop.shape[:2]
    if hh < 8 or ww < 8:
        return "unknown", 0.0

    crop = cv2.GaussianBlur(bgr_crop, (5, 5), 0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    assigned = np.zeros(h.shape, dtype=bool)
    counts = {}

    def add(name, mask):
        mask = mask & (~assigned)
        counts[name] = int(np.count_nonzero(mask))
        assigned[mask] = True

    add("black", v < 58)
    add("white", (s < 42) & (v >= 218))
    add("light_gray", (s < 55) & (v >= 150) & (v < 218))
    add("gray", (s < 58) & (v >= 95) & (v < 150))
    add("dark_gray", (s < 60) & (v >= 58) & (v < 95))

    add("beige", (h >= 8) & (h <= 28) & (s >= 18) & (s < 105) & (v >= 120))
    add("brown", (h >= 5) & (h <= 22) & (s >= 55) & (v >= 55) & (v < 170))

    chroma = (s >= 55) & (v >= 55)

    add("red", chroma & ((h <= 8) | (h >= 172)))
    add("orange", chroma & (h >= 9) & (h <= 22))
    add("yellow", chroma & (h >= 23) & (h <= 34))
    add("green", chroma & (h >= 35) & (h <= 85))
    add("cyan", chroma & (h >= 86) & (h <= 100))
    add("blue", chroma & (h >= 101) & (h <= 130))
    add("purple", chroma & (h >= 131) & (h <= 155))
    add("pink", chroma & (h >= 156) & (h <= 171))

    total = int(h.size)
    if total == 0 or not counts:
        return "unknown", 0.0

    name, count = max(counts.items(), key=lambda kv: kv[1])
    ratio = count / total

    if ratio < 0.20:
        return "unknown", ratio

    return name, ratio


def center_inside(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    cx = 0.5 * (ix1 + ix2)
    cy = 0.5 * (iy1 + iy2)

    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def pick_garment(garments, person_box, valid_types):
    p_area = max(box_area(person_box), 1.0)
    candidates = []

    for g in garments:
        if g["name"] not in valid_types:
            continue

        if not center_inside(g["box"], person_box):
            continue

        ratio = box_area(g["box"]) / p_area

        if ratio > 0.90:
            continue

        candidates.append(g)

    if not candidates:
        return None

    return max(candidates, key=lambda g: g["conf"])


def create_csi_pipeline(sensor_id=0, width=1280, height=720, fps=30):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
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
            "Check CSI/USB camera and OpenCV GStreamer/V4L2 support."
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


def load_model(model_path, use_world_prompts):
    model = YOLO(model_path)

    if use_world_prompts:
        model.set_classes(WORLD_CLASSES)

    return model


def parse_result(result):
    detections = []

    if result.boxes is None:
        return detections

    names = result.names

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)

    if result.boxes.id is not None:
        ids = result.boxes.id.detach().cpu().numpy().astype(int)
    else:
        ids = np.full(len(xyxy), -1, dtype=int)

    for box, score, cls_id, track_id in zip(
        xyxy,
        conf,
        cls,
        ids,
    ):
        detections.append(
            {
                "box": tuple(float(x) for x in box),
                "conf": float(score),
                "cls_id": int(cls_id),
                "name": str(names[int(cls_id)]),
                "track_id": int(track_id),
            }
        )

    return detections


def iso_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def today_str():
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def hhmmss(ts):
    try:
        return datetime.fromisoformat(ts).strftime("%H:%M:%S")
    except Exception:
        return ts


def seconds_between(a, b):
    try:
        dt_a = datetime.fromisoformat(a)
        dt_b = datetime.fromisoformat(b)
        return max(
            0,
            int((dt_b - dt_a).total_seconds()),
        )
    except Exception:
        return 0


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        tmp,
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
        tmp,
        path,
    )


def dominant_value(values, fallback="unknown"):
    valid = [
        v
        for v in values
        if v and v != "unknown"
    ]

    if not valid:
        return fallback, 0.0

    value, count = Counter(valid).most_common(1)[0]

    return (
        value,
        count / len(valid),
    )


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
        obs,
    ):
        self.upper_color.append(
            obs["upper"]["color"]
        )

        self.upper_type.append(
            obs["upper"]["type"]
        )

        self.lower_color.append(
            obs["lower"]["color"]
        )

        self.lower_type.append(
            obs["lower"]["type"]
        )


    def stable(self):
        uc, ucs = dominant_value(
            self.upper_color
        )

        ut, uts = dominant_value(
            self.upper_type,
            "top",
        )

        lc, lcs = dominant_value(
            self.lower_color
        )

        lt, lts = dominant_value(
            self.lower_type,
            "bottom",
        )

        return {
            "upper": {
                "type": ut,
                "color": uc,
                "color_ko": COLOR_KO.get(
                    uc,
                    uc,
                ),
                "stability": round(
                    min(
                        ucs,
                        uts,
                    ),
                    3,
                ),
            },

            "lower": {
                "type": lt,
                "color": lc,
                "color_ko": COLOR_KO.get(
                    lc,
                    lc,
                ),
                "stability": round(
                    min(
                        lcs,
                        lts,
                    ),
                    3,
                ),
            },

            "description": (
                f"{COLOR_KO.get(uc, uc)} {ut} / "
                f"{COLOR_KO.get(lc, lc)} {lt}"
            ),
        }


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

        self.data = self._load_or_create()

        self.next_id = self._next_id_number()


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

            except Exception:
                pass

        return {
            "date": self.date,
            "created_at": iso_now(),
            "updated_at": iso_now(),
            "visitors": [],
        }


    def _next_id_number(self):
        highest = 0

        for v in self.data["visitors"]:
            vid = str(
                v.get(
                    "visitor_id",
                    "",
                )
            )

            if (
                vid.startswith("P")
                and vid[1:].isdigit()
            ):
                highest = max(
                    highest,
                    int(vid[1:]),
                )

        return highest + 1


    def new_visitor_id(self):
        vid = f"P{self.next_id:04d}"

        self.next_id += 1

        return vid


    def get(
        self,
        visitor_id,
    ):
        return next(
            (
                v
                for v in self.data["visitors"]
                if v["visitor_id"] == visitor_id
            ),
            None,
        )


    def create(
        self,
        visitor_id,
        tracker_id,
        timestamp,
        appearance,
    ):
        rec = {
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
        }

        self.data["visitors"].append(
            rec
        )


    def update(
        self,
        visitor_id,
        timestamp,
        appearance,
    ):
        rec = self.get(
            visitor_id
        )

        if rec is None:
            return

        old_desc = (
            rec
            .get(
                "appearance",
                {},
            )
            .get(
                "description"
            )
        )

        rec["last_seen"] = timestamp

        rec["last_seen_time"] = hhmmss(
            timestamp
        )

        rec["duration_seconds"] = seconds_between(
            rec["first_seen"],
            timestamp,
        )

        rec["status"] = "present"

        rec["appearance"] = appearance

        if (
            appearance["description"]
            != old_desc
        ):
            rec[
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


    def save(self):
        self.data[
            "updated_at"
        ] = iso_now()

        atomic_write_json(
            self.path,
            self.data,
        )


def observe_person(
    frame,
    person,
    garments,
):
    h_img, w_img = frame.shape[:2]

    pbox = clamp_box(
        person["box"],
        w_img,
        h_img,
    )

    upper_g = pick_garment(
        garments,
        pbox,
        UPPER_TYPES,
    )

    lower_g = pick_garment(
        garments,
        pbox,
        LOWER_TYPES,
    )

    upper_box = (
        shrink_box(
            upper_g["box"],
            0.10,
        )
        if upper_g
        else body_part_box(
            pbox,
            "upper",
        )
    )

    lower_box = (
        shrink_box(
            lower_g["box"],
            0.10,
        )
        if lower_g
        else body_part_box(
            pbox,
            "lower",
        )
    )

    ux1, uy1, ux2, uy2 = clamp_box(
        upper_box,
        w_img,
        h_img,
    )

    lx1, ly1, lx2, ly2 = clamp_box(
        lower_box,
        w_img,
        h_img,
    )

    upper_color, upper_ratio = dominant_clothing_color(
        frame[
            uy1:uy2,
            ux1:ux2,
        ]
    )

    lower_color, lower_ratio = dominant_clothing_color(
        frame[
            ly1:ly2,
            lx1:lx2,
        ]
    )

    return {
        "person_box": pbox,

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
                upper_g["name"]
                if upper_g
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
                lower_g["name"]
                if lower_g
                else "bottom"
            ),

            "color": lower_color,

            "color_ratio": round(
                lower_ratio,
                3,
            ),
        },
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="src/models/YOLO/appearance_worldv2.engine",
    )

    parser.add_argument(
        "--base-world",
        action="store_true",
    )

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

    parser.add_argument(
        "--output-dir",
        default="src/output/visitor_logs",
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

    args = parser.parse_args()

    model = load_model(
        args.model,
        args.base_world,
    )

    cap = open_camera(
        args
    )

    log = DailyVisitorLog(
        args.output_dir
    )

    tracker_to_visitor = {}

    smoothers = {}

    last_seen_mono = {}

    last_save = 0.0

    print(
        f"Logging to: {log.path}"
    )

    print(
        "q: quit"
    )

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame = maybe_flip(
                frame,
                args.flip,
            )

            infer_kwargs = dict(
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
            )

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
                d
                for d in detections
                if d["name"] == "person"
            ]

            garments = [
                d
                for d in detections
                if d["name"] != "person"
            ]

            now_iso = iso_now()

            now_mono = time.monotonic()

            visible_tids = set()

            for person in persons:

                tid = int(
                    person["track_id"]
                )

                if tid < 0:
                    continue

                visible_tids.add(
                    tid
                )

                last_seen_mono[
                    tid
                ] = now_mono

                if (
                    tid
                    not in tracker_to_visitor
                ):
                    tracker_to_visitor[
                        tid
                    ] = log.new_visitor_id()

                    smoothers[
                        tid
                    ] = AppearanceSmoother(
                        args.appearance_window
                    )

                visitor_id = tracker_to_visitor[
                    tid
                ]

                obs = observe_person(
                    frame,
                    person,
                    garments,
                )

                smoothers[
                    tid
                ].update(
                    obs
                )

                stable = smoothers[
                    tid
                ].stable()

                if (
                    log.get(
                        visitor_id
                    )
                    is None
                ):
                    log.create(
                        visitor_id,
                        tid,
                        now_iso,
                        stable,
                    )

                else:
                    log.update(
                        visitor_id,
                        now_iso,
                        stable,
                    )

                px1, py1, px2, py2 = obs[
                    "person_box"
                ]

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
                    (
                        0,
                        255,
                        0,
                    ),
                    2,
                )

                label = (
                    f"{visitor_id} | "
                    f"{stable['upper']['color']} "
                    f"{stable['upper']['type']} / "
                    f"{stable['lower']['color']} "
                    f"{stable['lower']['type']}"
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
                    0.55,
                    (
                        0,
                        255,
                        0,
                    ),
                    2,
                    cv2.LINE_AA,
                )

            stale = []

            for tid, seen in list(
                last_seen_mono.items()
            ):

                if (
                    tid not in visible_tids
                    and now_mono - seen
                    >= args.leave_timeout
                ):

                    visitor_id = tracker_to_visitor.get(
                        tid
                    )

                    if visitor_id:

                        rec = log.get(
                            visitor_id
                        )

                        if (
                            rec
                            and rec["status"]
                            == "present"
                        ):
                            rec[
                                "status"
                            ] = "left"

                            rec[
                                "closed_reason"
                            ] = "not_seen"

                    stale.append(
                        tid
                    )

            for tid in stale:

                last_seen_mono.pop(
                    tid,
                    None,
                )

                tracker_to_visitor.pop(
                    tid,
                    None,
                )

                smoothers.pop(
                    tid,
                    None,
                )

            if (
                now_mono - last_save
                >= args.save_interval
            ):
                log.save()

                last_save = now_mono

            cv2.putText(
                frame,
                (
                    f"Today visitors: "
                    f"{len(log.data['visitors'])}"
                ),
                (
                    20,
                    32,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (
                    0,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Visitor Appearance Logger",
                frame,
            )

            if (
                cv2.waitKey(1)
                & 0xFF
                == ord("q")
            ):
                break

    finally:

        for visitor_id in tracker_to_visitor.values():

            rec = log.get(
                visitor_id
            )

            if (
                rec
                and rec["status"]
                == "present"
            ):
                rec[
                    "status"
                ] = "left"

                rec[
                    "closed_reason"
                ] = "monitor_stopped"

        log.save()

        cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
