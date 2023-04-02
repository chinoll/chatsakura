from .create_model import *
import uvicorn
from fastapi import FastAPI

app = FastAPI()
device, model, tokenizer = create_model()

@app.get("/chat")
async def chat(query: str, topk: int = 5, topp: float = 0.95, temperature: float = 1.0,max_length: int=2048):
    input_ids = tokenizer.encode(query, return_tensors='pt').to(device)
    output = model.generate(
        input_ids,
        do_sample=True,
        max_length=max_length,
        top_k=topk,
        top_p=topp,
        temperature=temperature
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    send_feedback(query,response)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("chatsakura_api:app", host="0.0.0.0", port=8080, reload=True)