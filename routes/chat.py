from fastapi import APIRouter, HTTPException
from schemas.chat import ChatRequest, ChatResponse
from utils.logger import logger

router = APIRouter(prefix="/chat",
                   tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Получен вопрос: {request.question}")

        return ChatResponse(
            answer="TEMP"
        )
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
