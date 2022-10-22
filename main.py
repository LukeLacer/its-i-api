from fastapi import FastAPI
from pydantic import BaseModel


class Image(BaseModel):
    imageData: str


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "The api is running"}


@app.post("/itsi")
def guessifitsi(data: Image):
    return {"message": "It can be an 'i'"}
