import gradio as gr
import requests
import os
from urllib.parse import quote
from .create_model import *
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
prompt = "Below is an <human request> that describes a task. Write a response that appropriately completes the request.lets think step-by-step.\n\n"

device, model, tokenizer = create_model()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    chatbot.style(height=500)
    msg = gr.Textbox(lines=5)
    with gr.Row():
        topk = gr.Slider(0, 100, value=5, label="TopK", info="TopK sample")
        temp = gr.Slider(0, 2, value=1, label="Temperature", info="Temperature")
        topp = gr.Slider(0, 1, value=0.95, label="TopP", info="TopP sample")

    with gr.Row():
        sumbit = gr.Button("sumbit")
        clear = gr.Button("Clear")
    def user(user_message, history,topk,temp,topp):
        text = f'{prompt}<human request>:{user_message}\n<bot response>:'
        tokens = tokenizer.encode(text,return_tensors="pt").to(device)
        outputs = model.generate(tokens,top_k=topk,top_p=topp,do_sample=True,temperature=temp,max_length=2048)
        print(outputs)
        bot_message = tokenizer.decode(outputs[0],skip_special_tokens=True)[len(text):]
        send_feedback(user_message, bot_message)
        return "", history + [[user_message, bot_message]]

    def bot(history):
        return history

    sumbit.click(user, [msg, chatbot,topk,temp,topp], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
