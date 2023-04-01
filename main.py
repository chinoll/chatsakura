import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import platform
import torch
prompt = "Below is an <human request> that describes a task. Write a response that appropriately completes the request.lets think step-by-step.\n\n"
device = None
def create_model():
    if platform.system() == 'Windows':
        default_type = 'fp16'
    else:
        default_type = 'int4'
    model_type = input(f"Selecting Model Types(fp16,int8,int4,default={default_type})") or default_type
    if torch.cuda.is_available():
        default_device = 'gpu'
    else:
        default_device = 'cpu'
    devices = input(f'Selecting device(gpu,cpu,default={default_device})') or default_device
    global device
    if devices == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if model_type == 'fp16':
        model_type = ''
    else:
        model_type = '-' + model_type
    model = AutoModelForCausalLM.from_pretrained(f"chinoll/chatsakura-3b{model_type}")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(f"chinoll/chatsakura-3b{model_type}")
    return model, tokenizer

model,tokenizer = create_model()
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
        # print(user_message,history)
        tokens = tokenizer.encode(f'{prompt}<human request>:{user_message}\n<bot response>:',return_tensors="pt").to(device)
        outputs = model.generate(tokens,topk=topk,top_p=topp,do_sample=True,temperature=temp)
        bot_message = tokenizer.decode(outputs[0],skip_special_tokens=True)[len(tokens):]
        return "", history + [[user_message, bot_message]]

    def bot(history):
        return history

    sumbit.click(user, [msg, chatbot,topk,temp,topp], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
