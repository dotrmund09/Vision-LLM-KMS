#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from llama_cpp import Llama


def today_str():
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def load_log(output_dir, date):
    path = Path(output_dir) / f"visitor_log_{date}.json"
    if not path.exists():
        return path, None
    with open(path, "r", encoding="utf-8") as f:
        return path, json.load(f)


def compact_visitors(data):
    rows = []
    for v in data.get("visitors", []):
        app = v.get("appearance", {})
        up = app.get("upper", {})
        low = app.get("lower", {})

        rows.append({
            "visitor_id": v.get("visitor_id"),
            "first_seen_time": v.get("first_seen_time"),
            "last_seen_time": v.get("last_seen_time"),
            "duration_seconds": v.get("duration_seconds"),
            "status": v.get("status"),
            "upper_color": up.get("color"),
            "upper_color_ko": up.get("color_ko"),
            "upper_type": up.get("type"),
            "lower_color": low.get("color"),
            "lower_color_ko": low.get("color_ko"),
            "lower_type": low.get("type"),
            "description": app.get("description"),
        })
    return rows


def build_prompt(question, date, visitors):
    records = json.dumps(visitors, ensure_ascii=False, indent=2)
    return f"""
조회 날짜: {date}

아래는 카메라 시스템이 저장한 익명 방문자 기록이다.
visitor_id는 실제 신원이 아니라 카메라 추적용 익명 ID다.

방문자 기록:
{records}

사용자 질문:
{question}

규칙:
1. 반드시 위 기록에 있는 내용만 사용해서 답한다.
2. 옷 색/종류 질문은 upper/lower 필드를 비교한다.
3. 시간 질문은 first_seen_time, last_seen_time을 사용한다.
4. 조건에 맞는 사람이 여러 명이면 visitor_id와 시간을 모두 알려준다.
5. 기록에 없으면 없다고 답한다.
6. 실제 신원, 성별, 나이 등 기록에 없는 사실은 추측하지 않는다.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf",
    )
    p.add_argument(
        "--output-dir",
        default="src/output/visitor_logs",
    )
    p.add_argument(
        "--date",
        default=today_str(),
        help="YYYY-MM-DD, default=today",
    )
    args = p.parse_args()

    path, data = load_log(args.output_dir, args.date)
    if data is None:
        print(f"기록 파일이 없습니다: {path}")
        return

    llm = Llama(
        model_path=args.model,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )

    print(f"Loaded: {path}")
    print(f"Visitors: {len(data.get('visitors', []))}")
    print("질문을 입력하세요. 종료: q")

    while True:
        question = input("\nYou> ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue

        # Always reload to use the most recent camera log.
        _, latest = load_log(args.output_dir, args.date)
        if latest is None:
            print("Gemma> 해당 날짜의 기록이 없습니다.")
            continue

        visitors = compact_visitors(latest)
        if not visitors:
            print("Gemma> 해당 날짜에 저장된 방문자 기록이 없습니다.")
            continue

        prompt = build_prompt(question, args.date, visitors)

        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 카메라 방문 기록 조회 도우미다. "
                        "제공된 JSON 데이터만 근거로 답한다. "
                        "visitor_id를 실제 사람의 신원으로 해석하지 않는다. "
                        "모르는 내용은 추측하지 않는다. "
                        "한국어로 간결하게 답한다."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=180,
            temperature=0.0,
        )

        answer = response["choices"][0]["message"]["content"].strip()
        print(f"Gemma> {answer}")


if __name__ == "__main__":
    main()
