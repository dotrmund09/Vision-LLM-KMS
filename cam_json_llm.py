# [ANSWER] 카메라 프레임 전체 화면의 YOLO 탐지 결과를 Dictionary → JSON → Natural Language → Gemma 순서로 전달

import cv2
import json

from ultralytics import YOLO
from llama_cpp import Llama


YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
JSON_PATH = "src/output/vision_data.json"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 150


def detections_to_text(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        vision_data = json.load(file)
        
    objects = vision_data["objects"]

    if len(objects) == 0:
        return "현재 탐지된 객체가 없습니다."

    sentences = []

    for index, obj in enumerate(objects, start=1):
        sentence = f"{index}번 객체는 {obj['class']}이며, confidence는 {obj['confidence']:.2f}입니다."

        sentences.append(sentence)

    return "\n".join(sentences)


yolo = YOLO(YOLO_MODEL_PATH)

llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), "
    "width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()


print("q : 종료")
print("l : 현재 YOLO 탐지 결과를 Gemma에게 전달")


while True:
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF

    if not ret:
        break
    if key == ord("q"):
        break

    height, width = frame.shape[:2]


    # 1. 현재 프레임 YOLO 탐지
    results = yolo.predict(
        source=frame,
        conf=0.25,
        iou=0.5,
        verbose=False,
    )

    result = results[0]


    # 2. YOLO 결과 화면 출력
    output_frame = result.plot()
    cv2.imshow("YOLO + Gemma", output_frame)


    # 3. YOLO 탐지 결과를 Dictionary 형태로 변환
    objects = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

        objects.append(
            {
                "class": result.names[class_id],
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                }
            }
        )

    vision_dict = {
        "image_width": width,
        "image_height": height,
        "objects": objects,
    }


    # 4. Dictionary → JSON 파일 저장
    if key == ord("l"):
        with open(JSON_PATH, "w", encoding="utf-8") as file:
            json.dump(vision_dict, file, ensure_ascii=False, indent=4)
        
        print("\n[JSON 파일 저장 완료]")

        # 5. JSON 파일 → Natural Language 변환
        vision_text = detections_to_text(JSON_PATH)
        print("\n[Vision Context]")
        print(vision_text)

        # 6. 탐지 결과를 Gemma에 Context로 전달
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": """
                                Instruction:
                                주어진 객체 탐지 정보를 바탕으로 현재 상황을 설명하시오.

                                Constraint:
                                탐지 결과에 없는 객체를 추측하지 마시오.

                                Output Format:
                                한국어 두 문장 이내.
                               """
                },
                {
                    "role": "user",
                    "content": f"""
                                Context:
                                {vision_text}
                               """
                },
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )

        answer = response["choices"][0]["message"]["content"]

        print("\n[Gemma]")
        print(answer)


cap.release()
cv2.destroyAllWindows()
