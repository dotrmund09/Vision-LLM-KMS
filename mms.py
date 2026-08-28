import cv2
import base64

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler


MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"
IMAGE_PATH = "src/images/city.png"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 150


chat_handler = Gemma4ChatHandler(clip_model_path=MMPROJ_PATH)

llm = Llama(
    model_path=MODEL_PATH,
    chat_handler=chat_handler,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)


image = cv2.imread(IMAGE_PATH)
success, buffer = cv2.imencode(".jpg", image)

if not success:
    raise RuntimeError("이미지 인코딩에 실패했습니다.")

image_base64 = base64.b64encode(buffer).decode("utf-8")
image_data = ("data:image/jpeg;base64," + image_base64)


response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": """
                        Instruction:
                        주어진 이미지를 바탕으로 현재 상황을 설명하시오.

                        Constraint:
                        이미지에서 명확하게 확인되지 않는 내용은 추측하지 마시오.

                        Output Format:
                        한국어 두 문장 이내.
                       """
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "현재 카메라 이미지를 설명하시오."
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
    temperature=0.7,
)


print(response["choices"][0]["message"]["content"])
