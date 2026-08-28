#!/usr/bin/env python3
"""
Visitor Voice Assistant
=======================

Camera/vision process (separate):
    visitor_monitor.py
        -> src/output/visitor_logs/visitor_log_YYYY-MM-DD.json

Voice assistant:
    Microphone
        -> arecord
        -> whisper.cpp STT
        -> latest visitor JSON reload
        -> Gemma
        -> Piper TTS
        -> Speaker

Default interaction:
    Press Enter
        -> record for 5 seconds
        -> ask Gemma
        -> speak answer
        -> repeat

This separation keeps the camera/YOLO process running independently
while Gemma/STT/TTS are used only when the user asks a question.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from llama_cpp import Llama


# ============================================================
# Time / JSON
# ============================================================

def today_str():
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def load_log(output_dir: str, date: str):
    path = Path(output_dir) / f"visitor_log_{date}.json"

    if not path.exists():
        return path, None

    with open(path, "r", encoding="utf-8") as f:
        return path, json.load(f)


def compact_visitors(data: dict):
    """
    Gemma에게 필요한 정보만 전달한다.
    원본 JSON의 불필요하게 긴 appearance_history는 제외한다.
    """
    rows = []

    for visitor in data.get("visitors", []):
        appearance = visitor.get("appearance", {})
        upper = appearance.get("upper", {})
        lower = appearance.get("lower", {})

        alerts = []
        for alert in visitor.get("alerts", []):
            alerts.append({
                "time": alert.get("time"),
                "event": alert.get("event"),
                "message": alert.get("message"),
            })

        rows.append({
            "visitor_id": visitor.get("visitor_id"),

            "first_seen_time": visitor.get("first_seen_time"),
            "last_seen_time": visitor.get("last_seen_time"),
            "duration_seconds": visitor.get("duration_seconds"),
            "status": visitor.get("status"),

            "upper": {
                "type": upper.get("type"),
                "color": upper.get("color"),
                "color_ko": upper.get("color_ko"),
            },

            "lower": {
                "type": lower.get("type"),
                "color": lower.get("color"),
                "color_ko": lower.get("color_ko"),
            },

            "description": appearance.get("description"),
            "alerts": alerts,
        })

    return rows


# ============================================================
# STT
# ============================================================

def record_audio(
    mic_device: str,
    audio_file: str,
    record_seconds: int,
):
    """
    ALSA arecord로 16 kHz / mono / S16_LE 녹음.

    기존 실습 코드처럼 pasuspender가 있으면 사용하고,
    없으면 arecord를 직접 실행한다.
    """
    audio_path = Path(audio_file)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    arecord_cmd = [
        "arecord",
        "-D", mic_device,
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "1",
        "-d", str(record_seconds),
        str(audio_path),
    ]

    if shutil.which("pasuspender"):
        cmd = [
            "pasuspender",
            "--",
            *arecord_cmd,
        ]
    else:
        cmd = arecord_cmd

    print(f"[MIC] {record_seconds}초 동안 말씀해 주세요...")

    subprocess.run(
        cmd,
        check=True,
    )

    return audio_path


def whisper_stt(
    whisper_path: str,
    whisper_model: str,
    audio_file: str,
):
    """
    whisper.cpp를 이용해 한국어 STT 수행.
    """
    result = subprocess.run(
        [
            whisper_path,
            "-m", whisper_model,
            "-f", audio_file,
            "-l", "ko",
            "--no-timestamps",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    text = result.stdout.strip()

    # whisper-cli 버전에 따라 일부 출력이 stderr에 나타나는 경우 대비
    if not text:
        text = result.stderr.strip()

    # 여러 줄인 경우 빈 줄 제거 후 합침
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # whisper 로그가 섞이는 환경에서도 마지막 자연어 줄을 우선 사용
    if not lines:
        return ""

    candidates = [
        line for line in lines
        if not line.startswith("whisper_")
        and not line.startswith("main:")
        and not line.startswith("system_info:")
    ]

    if candidates:
        text = " ".join(candidates)
    else:
        text = " ".join(lines)

    return text.strip()


def speech_to_text(args):
    record_audio(
        mic_device=args.mic_device,
        audio_file=args.input_audio,
        record_seconds=args.record_seconds,
    )

    question = whisper_stt(
        whisper_path=args.whisper_path,
        whisper_model=args.whisper_model,
        audio_file=args.input_audio,
    )

    return question


# ============================================================
# TTS
# ============================================================

def text_to_speech(
    text: str,
    piper_python: str,
    piper_model: str,
    output_file: str,
    speaker_device: str,
):
    """
    Piper로 답변 WAV 생성 -> aplay로 스피커 출력.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            piper_python,
            "-m", "piper",
            "-m", piper_model,
            "-f", str(output_path),
            "--",
            text,
        ],
        check=True,
    )

    subprocess.run(
        [
            "aplay",
            "-D", speaker_device,
            str(output_path),
        ],
        check=True,
    )


# ============================================================
# Gemma prompt
# ============================================================

def build_prompt(
    question: str,
    date: str,
    visitors: list,
):
    records = json.dumps(
        visitors,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
조회 날짜: {date}

아래 데이터는 카메라가 기록한 익명 방문자 기록이다.
visitor_id는 실제 신원이 아니라 카메라 추적용 익명 ID이다.

[방문자 기록]
{records}

[사용자 질문]
{question}

[답변 규칙]
1. 반드시 제공된 방문자 기록에 있는 내용만 사용한다.
2. 인상착의 질문은 upper와 lower의 type/color/color_ko를 이용한다.
3. 시간 질문은 first_seen_time과 last_seen_time을 이용한다.
4. 체류시간 질문은 duration_seconds를 이용한다.
5. 현재 화면에 있는지 물으면 status를 이용한다.
6. 경보 여부를 물으면 alerts를 이용한다.
7. 조건에 맞는 방문자가 여러 명이면 visitor_id와 시간을 함께 알려준다.
8. 기록에 없는 내용은 없다고 답하거나 확인할 수 없다고 답한다.
9. visitor_id를 실제 사람의 신원으로 해석하지 않는다.
10. 성별, 나이, 이름 등 저장되지 않은 정보는 추측하지 않는다.
11. 스피커로 읽기 쉽도록 한국어 1~3문장으로 간결하게 답한다.
"""


def ask_gemma(
    llm,
    question: str,
    date: str,
    visitors: list,
):
    prompt = build_prompt(
        question=question,
        date=date,
        visitors=visitors,
    )

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 카메라 방문 기록을 조회하는 음성 비서다. "
                    "제공된 JSON 데이터만 근거로 답한다. "
                    "기록에 없는 사실은 추측하지 않는다. "
                    "사용자가 듣기 편하도록 짧고 자연스러운 한국어로 답한다."
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

    return (
        response["choices"][0]["message"]["content"]
        .strip()
    )


# ============================================================
# Environment checks
# ============================================================

def check_path(path: str, description: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{description} 파일을 찾을 수 없습니다:\n{path}"
        )


def check_environment(args):
    check_path(
        args.gemma_model,
        "Gemma model",
    )

    check_path(
        args.whisper_path,
        "whisper-cli",
    )

    check_path(
        args.whisper_model,
        "Whisper model",
    )

    check_path(
        args.piper_python,
        "Piper Python",
    )

    check_path(
        args.piper_model,
        "Piper model",
    )

    if shutil.which("arecord") is None:
        raise RuntimeError(
            "arecord를 찾을 수 없습니다. alsa-utils 설치 여부를 확인하세요."
        )

    if shutil.which("aplay") is None:
        raise RuntimeError(
            "aplay를 찾을 수 없습니다. alsa-utils 설치 여부를 확인하세요."
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()


    # --------------------------------------------------------
    # Visitor JSON
    # --------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        default="src/output/visitor_logs",
    )

    parser.add_argument(
        "--date",
        default=today_str(),
        help="조회 날짜 YYYY-MM-DD, 기본값=오늘",
    )


    # --------------------------------------------------------
    # Gemma
    # --------------------------------------------------------

    parser.add_argument(
        "--gemma-model",
        default=(
            "src/models/Gemma4/"
            "google_gemma-4-E2B-it-Q4_K_M.gguf"
        ),
    )

    parser.add_argument(
        "--n-ctx",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=180,
    )


    # --------------------------------------------------------
    # Microphone / Whisper
    # --------------------------------------------------------

    parser.add_argument(
        "--mic-device",
        default="plughw:3,0",
    )

    parser.add_argument(
        "--input-audio",
        default="src/audio/input.wav",
    )

    parser.add_argument(
        "--record-seconds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--whisper-path",
        default=(
            "whisper.cpp/"
            "build-cpu/bin/"
            "whisper-cli"
        ),
    )

    parser.add_argument(
        "--whisper-model",
        default=(
            "whisper.cpp/models/"
            "ggml-base.bin"
        ),
    )


    # --------------------------------------------------------
    # Speaker / Piper
    # --------------------------------------------------------

    parser.add_argument(
        "--piper-python",
        default=".piper_venv/bin/python",
    )

    parser.add_argument(
        "--piper-model",
        default=(
            "src/models/Piper/"
            "ko_KR-kss-medium.onnx"
        ),
    )

    parser.add_argument(
        "--output-audio",
        default="src/audio/response.wav",
    )

    parser.add_argument(
        "--speaker-device",
        default="plughw:2,0",
    )

    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="답변을 스피커로 재생하지 않음",
    )


    args = parser.parse_args()


    # ========================================================
    # Check
    # ========================================================

    check_environment(args)


    # ========================================================
    # Load today's JSON
    # ========================================================

    json_path, data = load_log(
        args.output_dir,
        args.date,
    )

    if data is None:
        print(
            "[WARN] 아직 방문 기록 파일이 없습니다:"
        )
        print(json_path)
        print(
            "visitor_monitor.py를 먼저 실행해 "
            "기록을 생성하세요."
        )


    # ========================================================
    # Gemma load
    # ========================================================

    print(
        "[INFO] Gemma loading..."
    )

    llm = Llama(
        model_path=args.gemma_model,
        n_gpu_layers=-1,
        n_ctx=args.n_ctx,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )

    print(
        "[INFO] Gemma ready"
    )


    # ========================================================
    # Interaction
    # ========================================================

    print()
    print(
        "=============================================="
    )
    print(
        " Visitor Voice Assistant"
    )
    print(
        f" Date    : {args.date}"
    )
    print(
        f" JSON    : {json_path}"
    )
    print(
        f" Mic     : {args.mic_device}"
    )
    print(
        f" Speaker : {args.speaker_device}"
    )
    print(
        f" Record  : {args.record_seconds} sec"
    )
    print(
        "=============================================="
    )
    print()
    print(
        "Enter : 음성 질문"
    )
    print(
        "q + Enter : 종료"
    )
    print()


    while True:

        command = input(
            "[Enter=말하기 / q=종료] > "
        ).strip()

        if command.lower() in {
            "q",
            "quit",
            "exit",
        }:
            break


        # ====================================================
        # 1. STT
        # ====================================================

        try:

            question = speech_to_text(
                args
            )

        except subprocess.CalledProcessError as e:

            print(
                "[STT ERROR]",
                e,
            )

            continue

        except Exception as e:

            print(
                "[MIC/STT ERROR]",
                e,
            )

            continue


        if not question:

            print(
                "[STT] 음성을 인식하지 못했습니다."
            )

            continue


        print()
        print(
            f"You > {question}"
        )


        # ====================================================
        # 2. 최신 JSON 다시 읽기
        # ====================================================

        json_path, latest = load_log(
            args.output_dir,
            args.date,
        )

        if latest is None:

            answer = (
                f"{args.date}의 방문 기록이 없습니다."
            )

        else:

            visitors = compact_visitors(
                latest
            )

            if not visitors:

                answer = (
                    f"{args.date}에 저장된 "
                    "방문자 기록이 없습니다."
                )

            else:

                # ============================================
                # 3. Gemma
                # ============================================

                try:

                    answer = ask_gemma(
                        llm=llm,
                        question=question,
                        date=args.date,
                        visitors=visitors,
                    )

                except Exception as e:

                    print(
                        "[LLM ERROR]",
                        e,
                    )

                    continue


        print(
            f"Gemma > {answer}"
        )
        print()


        # ====================================================
        # 4. TTS
        # ====================================================

        if not args.no_tts:

            try:

                text_to_speech(
                    text=answer,
                    piper_python=args.piper_python,
                    piper_model=args.piper_model,
                    output_file=args.output_audio,
                    speaker_device=args.speaker_device,
                )

            except subprocess.CalledProcessError as e:

                print(
                    "[TTS ERROR]",
                    e,
                )

            except Exception as e:

                print(
                    "[TTS ERROR]",
                    e,
                )


    print(
        "Voice assistant 종료"
    )


if __name__ == "__main__":
    main()
