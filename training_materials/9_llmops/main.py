from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import time, logging
from dotenv import load_dotenv; 

load_dotenv()

app = FastAPI(title="TechStore Support API")
client = AsyncOpenAI()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    latency_ms: float
    tokens_used: int

SYSTEM_PROMPT = """You are a helpful TechStore support agent.
Answer questions about products, orders, and returns.
If you don't know, say so. Never make up policies."""

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.monotonic()
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.message}
            ],
            max_tokens=500,
            temperature=0.3,
        )
        latency_ms = (time.monotonic() - start) * 1000
        response_text = completion.choices[0].message.content
        tokens = completion.usage.total_tokens
        
        # Log for monitoring
        logging.info({"event": "chat", "session": req.session_id,
                      "latency_ms": round(latency_ms), "tokens": tokens})
        
        return ChatResponse(response=response_text,
                            latency_ms=round(latency_ms),
                            tokens_used=tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model": "gpt-4o-mini"}

@app.get("/")
async def root():
    return {"service": "TechStore Support API", "docs": "/docs"}