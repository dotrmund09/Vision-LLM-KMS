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

from appearance_detector import (
    LOWER_TYPES, UPPER_TYPES, body_part_box, clamp_box, load_model,
    maybe_flip, open_camera, parse_result, pick_garment, shrink_box,
)
from appearance_monitor import COLOR_KO, dominant_clothing_color


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
        return max(0, int((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()))
    except Exception:
        return 0


def atomic_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def dominant_value(values, fallback="unknown"):
    valid = [v for v in values if v and v != "unknown"]
    if not valid:
        return fallback, 0.0
    value, count = Counter(valid).most_common(1)[0]
    return value, count / len(valid)


class AppearanceSmoother:
    def __init__(self, window=15):
        self.upper_color = deque(maxlen=window)
        self.upper_type = deque(maxlen=window)
        self.lower_color = deque(maxlen=window)
        self.lower_type = deque(maxlen=window)

    def update(self, obs):
        self.upper_color.append(obs["upper"]["color"])
        self.upper_type.append(obs["upper"]["type"])
        self.lower_color.append(obs["lower"]["color"])
        self.lower_type.append(obs["lower"]["type"])

    def stable(self):
        uc, ucs = dominant_value(self.upper_color)
        ut, uts = dominant_value(self.upper_type, "top")
        lc, lcs = dominant_value(self.lower_color)
        lt, lts = dominant_value(self.lower_type, "bottom")
        return {
            "upper": {
                "type": ut,
                "color": uc,
                "color_ko": COLOR_KO.get(uc, uc),
                "stability": round(min(ucs, uts), 3),
            },
            "lower": {
                "type": lt,
                "color": lc,
                "color_ko": COLOR_KO.get(lc, lc),
                "stability": round(min(lcs, lts), 3),
            },
            "description": f"{COLOR_KO.get(uc, uc)} {ut} / {COLOR_KO.get(lc, lc)} {lt}",
        }


class DailyVisitorLog:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.date = today_str()
        self.path = self.output_dir / f"visitor_log_{self.date}.json"
        self.data = self._load_or_create()
        self.next_id = self._next_id_number()

        changed = False
        for v in self.data["visitors"]:
            if v.get("status") == "present":
                v["status"] = "left"
                v["closed_reason"] = "monitor_restarted"
                changed = True
        if changed:
            self.save()

    def _load_or_create(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
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
            vid = str(v.get("visitor_id", ""))
            if vid.startswith("P") and vid[1:].isdigit():
                highest = max(highest, int(vid[1:]))
        return highest + 1

    def new_visitor_id(self):
        vid = f"P{self.next_id:04d}"
        self.next_id += 1
        return vid

    def get(self, visitor_id):
        return next((v for v in self.data["visitors"] if v["visitor_id"] == visitor_id), None)

    def create(self, visitor_id, tracker_id, timestamp, appearance):
        rec = {
            "visitor_id": visitor_id,
            "tracker_id": int(tracker_id),
            "first_seen": timestamp,
            "last_seen": timestamp,
            "first_seen_time": hhmmss(timestamp),
            "last_seen_time": hhmmss(timestamp),
            "duration_seconds": 0,
            "status": "present",
            "appearance": appearance,
            "appearance_history": [{
                "timestamp": timestamp,
                "description": appearance["description"],
                "upper": appearance["upper"],
                "lower": appearance["lower"],
            }],
        }
        self.data["visitors"].append(rec)

    def update(self, visitor_id, timestamp, appearance):
        rec = self.get(visitor_id)
        if rec is None:
            return
        old_desc = rec.get("appearance", {}).get("description")
        rec["last_seen"] = timestamp
        rec["last_seen_time"] = hhmmss(timestamp)
        rec["duration_seconds"] = seconds_between(rec["first_seen"], timestamp)
        rec["status"] = "present"
        rec["appearance"] = appearance

        if appearance["description"] != old_desc:
            rec["appearance_history"].append({
                "timestamp": timestamp,
                "description": appearance["description"],
                "upper": appearance["upper"],
                "lower": appearance["lower"],
            })

    def save(self):
        self.data["updated_at"] = iso_now()
        atomic_write_json(self.path, self.data)


def observe_person(frame, person, garments):
    h_img, w_img = frame.shape[:2]
    pbox = clamp_box(person["box"], w_img, h_img)

    upper_g = pick_garment(garments, pbox, UPPER_TYPES)
    lower_g = pick_garment(garments, pbox, LOWER_TYPES)

    upper_box = shrink_box(upper_g["box"], 0.10) if upper_g else body_part_box(pbox, "upper")
    lower_box = shrink_box(lower_g["box"], 0.10) if lower_g else body_part_box(pbox, "lower")

    ux1, uy1, ux2, uy2 = clamp_box(upper_box, w_img, h_img)
    lx1, ly1, lx2, ly2 = clamp_box(lower_box, w_img, h_img)

    upper_color, upper_ratio = dominant_clothing_color(frame[uy1:uy2, ux1:ux2])
    lower_color, lower_ratio = dominant_clothing_color(frame[ly1:ly2, lx1:lx2])

    return {
        "person_box": pbox,
        "upper_box": (ux1, uy1, ux2, uy2),
        "lower_box": (lx1, ly1, lx2, ly2),
        "upper": {
            "type": upper_g["name"] if upper_g else "top",
            "color": upper_color,
            "color_ratio": round(upper_ratio, 3),
        },
        "lower": {
            "type": lower_g["name"] if lower_g else "bottom",
            "color": lower_color,
            "color_ratio": round(lower_ratio, 3),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="src/models/YOLO/appearance_worldv2.engine")
    p.add_argument("--base-world", action="store_true")
    p.add_argument("--camera", choices=["csi", "usb"], default="csi")
    p.add_argument("--sensor-id", type=int, default=0)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--flip", choices=["none", "horizontal", "vertical", "both"], default="none")
    p.add_argument("--output-dir", default="src/output/visitor_logs")
    p.add_argument("--appearance-window", type=int, default=15)
    p.add_argument("--leave-timeout", type=float, default=3.0)
    p.add_argument("--save-interval", type=float, default=1.0)
    args = p.parse_args()

    model = load_model(args.model, args.base_world)
    cap = open_camera(args)
    log = DailyVisitorLog(args.output_dir)

    tracker_to_visitor = {}
    smoothers = {}
    last_seen_mono = {}
    last_save = 0.0

    print(f"Logging to: {log.path}")
    print("q: quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = maybe_flip(frame, args.flip)
            infer_kwargs = dict(
                imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False
            )
            if not str(args.model).endswith(".engine"):
                infer_kwargs["device"] = 0

            result = model.track(
                frame, persist=True, tracker="bytetrack.yaml", **infer_kwargs
            )[0]

            detections = parse_result(result)
            persons = [d for d in detections if d["name"] == "person"]
            garments = [d for d in detections if d["name"] != "person"]

            now_iso = iso_now()
            now_mono = time.monotonic()
            visible_tids = set()

            for person in persons:
                tid = int(person["track_id"])
                if tid < 0:
                    continue

                visible_tids.add(tid)
                last_seen_mono[tid] = now_mono

                if tid not in tracker_to_visitor:
                    tracker_to_visitor[tid] = log.new_visitor_id()
                    smoothers[tid] = AppearanceSmoother(args.appearance_window)

                visitor_id = tracker_to_visitor[tid]
                obs = observe_person(frame, person, garments)
                smoothers[tid].update(obs)
                stable = smoothers[tid].stable()

                if log.get(visitor_id) is None:
                    log.create(visitor_id, tid, now_iso, stable)
                else:
                    log.update(visitor_id, now_iso, stable)

                px1, py1, px2, py2 = obs["person_box"]
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                label = (
                    f"{visitor_id} | {stable['upper']['color']} {stable['upper']['type']} / "
                    f"{stable['lower']['color']} {stable['lower']['type']}"
                )
                cv2.putText(
                    frame, label, (px1, max(25, py1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA
                )

            stale = []
            for tid, seen in list(last_seen_mono.items()):
                if tid not in visible_tids and now_mono - seen >= args.leave_timeout:
                    visitor_id = tracker_to_visitor.get(tid)
                    if visitor_id:
                        rec = log.get(visitor_id)
                        if rec and rec["status"] == "present":
                            rec["status"] = "left"
                            rec["closed_reason"] = "not_seen"
                    stale.append(tid)

            for tid in stale:
                last_seen_mono.pop(tid, None)
                tracker_to_visitor.pop(tid, None)
                smoothers.pop(tid, None)

            if now_mono - last_save >= args.save_interval:
                log.save()
                last_save = now_mono

            cv2.putText(
                frame, f"Today visitors: {len(log.data['visitors'])}",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (0, 255, 255), 2, cv2.LINE_AA
            )
            cv2.imshow("Visitor Appearance Logger", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        for visitor_id in tracker_to_visitor.values():
            rec = log.get(visitor_id)
            if rec and rec["status"] == "present":
                rec["status"] = "left"
                rec["closed_reason"] = "monitor_stopped"
        log.save()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
