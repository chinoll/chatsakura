# chatSakura: Open-source multilingual conversational model.
chatsakura is a large multilingual conversational model based on Bloomz, supporting Chinese, English, Japanese, German, and French.<br>
The model size is only 3B.

# Model Limitations
The current limitations of the model include:

1. The model's ability in some scenarios involving mathematical reasoning, code, etc., still needs improvement.
2. The model cannot identify harmful instructions.

# Model Release
| Model precision| FP16 | int8 | int4 |
| ----- | ----- | ----- | ----- |
| Finetuned Model | [chatSakura-3b](https://huggingface.co/chinoll/chatsakura-3b) | [chatSakura-3b-int8](https://huggingface.co/chinoll/chatsakura-3b-int8) | [chatSakura-3b-int4](https://huggingface.co/chinoll/chatsakura-3b-int4) |

# Installation and Usage
Note: int4 and int8 precision must be run on a GPU and have not been tested on Windows yet.
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
python main.py
```

# Hardware Requirements
| Model precision| FP16 | int8 | int4 |
| ----- | ----- | ----- | ----- |
| Finetuned Model | 10G | 6G | 4G(recommended 6G) |