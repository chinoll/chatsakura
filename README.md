# chatSakura:Open-source multilingual conversational model.
[EN](https://github.com/chinoll/chatsakura/blob/master/README_EN.md)<br>
chatsakura是一个基于bloomz的多语言对话大模型，支持中文、英语、日语、德语、法语。<br>
模型大小仅为3B。

# 模型局限性
该模型目前存在以下问题：
1. 在一些涉及数理推理、代码等场景下模型的能力仍有待提高。
2. 无法鉴别危害性指令
   
# 模型发布
| Model precision| FP16 | int8 | int4 |
| ----- | ----- | ----- | ----- |
| Finetuned Model | [chatSakura-3b](https://huggingface.co/chinoll/chatsakura-3b) | [chatSakura-3b-int8](https://huggingface.co/chinoll/chatsakura-3b-int8) | [chatSakura-3b-int4](https://huggingface.co/chinoll/chatsakura-3b-int4) |

# 安装使用
注意：int4和int8精度必须在GPU上运行，在Windows下尚未测试过是否可以运行。
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
python main.py
```