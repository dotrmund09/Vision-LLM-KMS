# TODO: JSON 파일을 사용하여 LLM에 context 전달


from llama_cpp import Llama


MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
CONTEXT_WINDOW = 2048
MAX_TOKENS = 150

vision_text = detections_to_text("src/output/vision_data.json")

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)

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
        }
    ],
    max_tokens=MAX_TOKENS,
    temperature=0.7
)


print(response["choices"][0]["message"]["content"])