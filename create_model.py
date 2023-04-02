import platform
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import requests
def create_model():
    if platform.system() == 'Windows':
        default_type = 'fp16'
    else:
        default_type = 'int4'
    model_type = input(f"Selecting Model Types(fp16,int8,int4,default={default_type}):") or default_type

    if torch.cuda.is_available():
        default_device = 'gpu'
    else:
        default_device = 'cpu'
    devices = input(f'Selecting device(gpu,cpu,default={default_device}):') or default_device
    if devices == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if model_type == 'fp16':
        model_type = ''
    else:
        model_type = '-' + model_type

    model = AutoModelForCausalLM.from_pretrained(f"chinoll/chatsakura-3b{model_type}",trust_remote_code=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(f"chinoll/chatsakura-3b{model_type}")
    return device, model, tokenizer

def send_feedback(instruction, response):
    try:
        print("send feedback")
        # bypass GFW
        requests.get(f"http://162.159.136.129?question={quote(instruction)}&answer={quote(response)}",timeout=5,headers={'Host':'openchat.chinoll.org'},allow_redirects=True, verify=False)
    except Exception as e:
        print(e)