import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class Values(BaseModel):
    Personality: float
    Vision: float
    Politics: float
    Culture: float
    Beliefs: float

class RequestModel(BaseModel):
    name: str
    Values: Values

@app.post("/model")
async def create_item(request: RequestModel):
    return {"message": "Hello, World"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
