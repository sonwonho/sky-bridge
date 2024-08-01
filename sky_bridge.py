import base64

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from gooroom import Gooroom
from qnawithrag import QnARag


class GooRoomRequest(BaseModel):
    year: int = None
    subject: str = None
    grade: int = None
    university: str = None


app = FastAPI()
gr = Gooroom()
rag = QnARag()


@app.get("/chat/{query}")
async def chat(query: str):
    def to_stream(query):
        for r in rag.rag(base64.b64decode(query).decode("UTF-8")):
            yield r

    return StreamingResponse(to_stream(query), media_type="text/plain")


@app.get("/gooroom")
async def chat(request: GooRoomRequest = Depends()):
    result = gr.ask_schedule(
        year=request.year,
        subject=request.subject,
        grade=request.grade,
        university=request.university,
    )
    return JSONResponse(result)
