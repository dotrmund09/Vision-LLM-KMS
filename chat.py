from llama_cpp import Llama


MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
CONTEXT_WINDOW = 2048
MAX_TOKENS = 300


llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)


print("Gemma 4 Local Chatbot")
print("종료하려면 exit, quit, q, 종료 중 하나를 입력하세요.\n")


while True:
    user_input = input("무엇을 도와드릴까요?")

    if user_input.lower() in ["exit", "quit", "q", "종료"]:
        break

    messages = [
        {
            "role": "user",
            "content": user_input
        }
    ]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
    )

    answer = response["choices"][0]["message"]["content"]

    print(answer)