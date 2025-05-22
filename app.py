import gradio as gr
from huggingface_hub import AsyncInferenceClient
from openai import AsyncOpenAI


async def hf_chat(message, history, api_key: str = "") -> str:
    """
    使用 Hugging Face 的聊天模型進行對話。
    """
    if not api_key:
        raise gr.Error("API key is required for Hugging Face Inference API.")

    hf_client = AsyncInferenceClient(
        provider="fireworks-ai",
        api_key=api_key,
    )
    response = await hf_client.chat_completion(
        model="Qwen/Qwen3-30B-A3B",
        messages=history + [{"role": "user", "content": message}],
    )  # type: ignore
    return response.choices[0].message.content


async def oai_chat(message, history, api_key: str = "") -> str:
    if not api_key:
        raise gr.Error("API key is required for OpenAI API.")

    oai_client = AsyncOpenAI(api_key=api_key)
    response = await oai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=history + [{"role": "user", "content": message}],
    )  # type: ignore
    return response.choices[0].message.content


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Image("img/llama_with_hats.png")
    with gr.Row():
        with gr.Column():
            chat = gr.ChatInterface(
                fn=hf_chat,  # 使用 Hugging Face 的聊天模型
                title="Chat with LLM",  # 標題
                type="messages",
                additional_inputs=[
                    gr.Textbox(
                        label="Hugging Face API Key",
                        placeholder="Enter your Hugging Face API key here",
                        type="password",
                        value="",
                    ),
                ],
            )

demo.launch(share=True, debug=True)  # 啟動 Gradio 介面，並分享連
