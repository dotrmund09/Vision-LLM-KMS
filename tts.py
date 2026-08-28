import subprocess


PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = "src/audio/response.wav"
SPEAKER_DEVICE = "plughw:2,0"


text = "현재는 빨간불입니다. 주의하시길 바랍니다!"

subprocess.run(
    [
        PIPER_PYTHON,
        "-m",
        "piper",

        "-m",
        PIPER_MODEL,

        "-f",
        OUTPUT_FILE,

        "--",
        text,
    ],
    check=True,
)

subprocess.run(
    [
        "aplay",
        "-D",
        SPEAKER_DEVICE,
        OUTPUT_FILE,
    ],
    check=True,
)