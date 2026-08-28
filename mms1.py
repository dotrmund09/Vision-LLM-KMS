import cv2
import base64

from ultralytics import YOLO
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler


YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"
IMAGE_PATH = "src/images/city.png"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 20
DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600


yolo = YOLO(YOLO_MODEL_PATH)

chat_handler = Gemma4ChatHandler(clip_model_path=MMPROJ_PATH)

llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    chat_handler=chat_handler,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)


image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError("이미지를 불러올 수 없습니다.")


results = yolo.predict(
    source=image,
    conf=0.015,
    iou=0.5,
    classes=[9],
    verbose=False,
)

result = results[0]


if len(result.boxes) == 0:
    raise RuntimeError("신호등이 탐지되지 않았습니다.")


selected_box = min(result.boxes, key=lambda box: float(box.xyxy[0][1].item()))
confidence = float(selected_box.conf[0].item())
x1, y1, x2, y2 = (selected_box.xyxy[0].cpu().tolist())

x1 = int(x1)
y1 = int(y1)
x2 = int(x2)
y2 = int(y2)


height, width = image.shape[:2]

x1 = max(0, x1)
y1 = max(0, y1)
x2 = min(width, x2)
y2 = min(height, y2)

traffic_light_image = image[y1:y2, x1:x2].copy()

if traffic_light_image.size == 0:
    raise RuntimeError("BBox 이미지를 만들 수 없습니다.")


display_image = cv2.resize(traffic_light_image, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)

cv2.namedWindow("Traffic Light", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Light", DISPLAY_WIDTH, DISPLAY_HEIGHT)
cv2.imshow("Traffic Light", display_image)

while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyWindow("Traffic Light")


success, buffer = cv2.imencode(".jpg", traffic_light_image)

if not success:
    raise RuntimeError("이미지 인코딩에 실패했습니다.")

image_base64 = base64.b64encode(buffer).decode("utf-8")
image_data = "data:image/jpeg;base64," + image_base64


response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": """
                        Instruction:
                        주어진 신호등 이미지의 현재 색을 판단하시오.

                        Constraint:
                        반드시 다음 세 가지 중 하나만 대답하시오.
                        빨간불
                        노란불
                        파란불

                        다른 설명이나 문장을 추가하지 마시오.

                        Output Format:
                        한 단어.
                       """
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "현재 신호등의 색을 판단하시오."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data
                    },
                },
            ],
        }
    ],
    max_tokens=MAX_TOKENS,
    temperature=0.0,
)


answer = response["choices"][0]["message"]["content"].strip()

print("\n[Gemma]")
print(answer)
