#!/usr/bin/env python3
"""
Registered appearance monitor for Jetson.

The registered profile is an APPEARANCE profile, not an identity.
Example:
    upper = light_gray
    lower = black

Pipeline:
    Camera
      -> YOLO-World + ByteTrack
      -> upper/lower ROI
      -> HSV color
      -> temporal confirmation
      -> src/output/appearance_alert.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from appearance_detector import (
    LOWER_TYPES,
    UPPER_TYPES,
    body_part_box,
    clamp_box,
    load_model,
    maybe_flip,
    open_camera,
    parse_result,
    pick_garment,
    shrink_box,
)


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


def load_profile(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required = ["profile_id", "upper", "lower", "confirm_window", "confirm_hits", "cooldown_seconds"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Target profile missing keys: {missing}")

    if cfg["confirm_hits"] > cfg["confirm_window"]:
        raise ValueError("confirm_hits must be <= confirm_window")

    return cfg


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(tmp, path)


def dominant_clothing_color(bgr_crop):
    """
    Clothing-oriented HSV classifier.

    Important:
    - achromatic colors are separated mainly by S and V.
    - light_gray is intentionally separated from white/gray for the target profile.
    - thresholds should be calibrated for the actual camera + lighting.
    """
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

    # Achromatic colors first.
    add("black", v < 58)

    # White: low saturation + very high brightness.
    add("white", (s < 42) & (v >= 218))

    # Light gray: low saturation + high brightness, but below white.
    add("light_gray", (s < 55) & (v >= 150) & (v < 218))

    # Mid / dark gray.
    add("gray", (s < 58) & (v >= 95) & (v < 150))
    add("dark_gray", (s < 60) & (v >= 58) & (v < 95))

    # Warm low-saturation colors.
    add("beige", (h >= 8) & (h <= 28) & (s >= 18) & (s < 105) & (v >= 120))
    add("brown", (h >= 5) & (h <= 22) & (s >= 55) & (v >= 55) & (v < 170))

    # Chromatic colors.
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

    # Mixed / background-heavy ROI.
    if ratio < 0.20:
        return "unknown", ratio

    return name, ratio


def type_matches(observed, allowed):
    """Empty allowed list means clothing type is not part of the condition."""
    if not allowed:
        return True
    return observed in allowed


def appearance_matches(observed, target):
    upper_ok = (
        observed["upper"]["color"] == target["upper"]["color"]
        and type_matches(observed["upper"]["type"], target["upper"].get("types", []))
    )
    lower_ok = (
        observed["lower"]["color"] == target["lower"]["color"]
        and type_matches(observed["lower"]["type"], target["lower"].get("types", []))
    )
    return upper_ok and lower_ok


def save_alert_crop(frame, person_box, out_path):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(person_box, w, h)
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = out_path.with_name(out_path.stem + "_tmp" + out_path.suffix)
    if not cv2.imwrite(str(temp_path), crop):
        return None

    os.replace(temp_path, out_path)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="target_profile.json")
    parser.add_argument("--model", default="src/models/YOLO/appearance_worldv2.engine")
    parser.add_argument("--base-world", action="store_true")

    parser.add_argument("--camera", choices=["csi", "usb"], default="csi")
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--flip", choices=["none", "horizontal", "vertical", "both"], default="none")

    parser.add_argument("--alert-json", default="src/output/appearance_alert.json")
    parser.add_argument("--alert-image", default="src/output/appearance_alert.jpg")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    model = load_model(args.model, args.base_world)
    cap = open_camera(args)

    confirm_window = int(profile["confirm_window"])
    confirm_hits = int(profile["confirm_hits"])
    cooldown = float(profile["cooldown_seconds"])

    match_history = defaultdict(lambda: deque(maxlen=confirm_window))
    last_seen = {}
    last_alert = defaultdict(lambda: -1e9)

    print("=== Registered appearance monitor ===")
    print("Profile:", profile["profile_id"])
    print("Description:", profile.get("description", ""))
    print(
        f"Confirmation: {confirm_hits}/{confirm_window} frames, "
        f"cooldown={cooldown:.1f}s"
    )
    print("q: quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = maybe_flip(frame, args.flip)
            h_img, w_img = frame.shape[:2]

            infer_kwargs = dict(
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
            )

            if not str(args.model).endswith(".engine"):
                infer_kwargs["device"] = 0

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                **infer_kwargs,
            )[0]

            detections = parse_result(result)
            persons = [d for d in detections if d["name"] == "person"]
            garments = [d for d in detections if d["name"] != "person"]

            now = time.time()
            visible_ids = set()

            for person in persons:
                track_id = int(person["track_id"])

                # Do not merge multiple people into one history before tracker ID exists.
                if track_id < 0:
                    continue

                visible_ids.add(track_id)
                last_seen[track_id] = now

                pbox = clamp_box(person["box"], w_img, h_img)
                px1, py1, px2, py2 = pbox

                upper_g = pick_garment(garments, pbox, UPPER_TYPES)
                lower_g = pick_garment(garments, pbox, LOWER_TYPES)

                upper_box = (
                    shrink_box(upper_g["box"], 0.10)
                    if upper_g
                    else body_part_box(pbox, "upper")
                )
                lower_box = (
                    shrink_box(lower_g["box"], 0.10)
                    if lower_g
                    else body_part_box(pbox, "lower")
                )

                ux1, uy1, ux2, uy2 = clamp_box(upper_box, w_img, h_img)
                lx1, ly1, lx2, ly2 = clamp_box(lower_box, w_img, h_img)

                upper_color, upper_ratio = dominant_clothing_color(frame[uy1:uy2, ux1:ux2])
                lower_color, lower_ratio = dominant_clothing_color(frame[ly1:ly2, lx1:lx2])

                upper_type = upper_g["name"] if upper_g else "top"
                lower_type = lower_g["name"] if lower_g else "bottom"

                observed = {
                    "track_id": track_id,
                    "person_confidence": round(person["conf"], 3),
                    "bbox_xyxy": [px1, py1, px2, py2],
                    "upper": {
                        "type": upper_type,
                        "color": upper_color,
                        "color_ko": COLOR_KO.get(upper_color, upper_color),
                        "color_ratio": round(upper_ratio, 3),
                    },
                    "lower": {
                        "type": lower_type,
                        "color": lower_color,
                        "color_ko": COLOR_KO.get(lower_color, lower_color),
                        "color_ratio": round(lower_ratio, 3),
                    },
                }

                current_match = appearance_matches(observed, profile)
                match_history[track_id].append(1 if current_match else 0)

                hits = sum(match_history[track_id])
                samples = len(match_history[track_id])
                confirmed = samples >= confirm_window and hits >= confirm_hits

                # Visualization
                box_color = (0, 0, 255) if confirmed else (0, 255, 0)
                cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
                cv2.rectangle(frame, (ux1, uy1), (ux2, uy2), (255, 200, 0), 1)
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 0, 200), 1)

                text = (
                    f"ID {track_id}: {upper_color} {upper_type} / "
                    f"{lower_color} {lower_type}  match={hits}/{samples}"
                )
                cv2.putText(
                    frame,
                    text,
                    (px1, max(25, py1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2,
                    cv2.LINE_AA,
                )

                if confirmed and (now - last_alert[track_id] >= cooldown):
                    last_alert[track_id] = now

                    image_path = save_alert_crop(frame, pbox, args.alert_image)

                    event = {
                        "event": "registered_appearance_match",
                        "timestamp": now,
                        "profile": {
                            "profile_id": profile["profile_id"],
                            "description": profile.get("description", ""),
                            "upper": profile["upper"],
                            "lower": profile["lower"],
                        },
                        "observed": observed,
                        "confirmation": {
                            "hits": hits,
                            "window": confirm_window,
                            "required_hits": confirm_hits,
                        },
                        "alert_image": image_path,
                        "identity_confirmed": False,
                    }

                    atomic_write_json(args.alert_json, event)

                    print(
                        f"[ALERT EVENT] ID={track_id}, "
                        f"{observed['upper']['color_ko']} {upper_type} + "
                        f"{observed['lower']['color_ko']} {lower_type}"
                    )

            # Remove stale tracker history after 5 seconds out of view.
            stale_ids = [
                tid for tid, seen_t in last_seen.items()
                if now - seen_t > 5.0
            ]
            for tid in stale_ids:
                match_history.pop(tid, None)
                last_seen.pop(tid, None)

            target_text = (
                f"Target: {COLOR_KO.get(profile['upper']['color'], profile['upper']['color'])} upper + "
                f"{COLOR_KO.get(profile['lower']['color'], profile['lower']['color'])} lower"
            )
            cv2.putText(
                frame,
                target_text,
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Registered Appearance Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
