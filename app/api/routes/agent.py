from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.agent.agent import process_user_query

router = APIRouter()

@router.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    user_query = data.get("query", "")
    context = data.get("context", "")

    # Get the reply (normal)
    output = process_user_query(user_query, user_id="anonymous", context=context)
    reply = output.get("reply", "Sorry, something went wrong.")

    # STREAMING: send raw word tokens with trailing space
    def stream():
        for word in reply.split():
            yield f"data: {word} \n\n"  
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
