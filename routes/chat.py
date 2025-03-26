from fastapi import APIRouter, HTTPException, Request
from schemas.chat import ChatRequest, ChatResponse
from utils.logger import logger
from langchain_core.runnables import RunnableConfig
from config import settings

router = APIRouter(prefix="/chat",
                   tags=["chat"])

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, fastapi_request: Request):
    try:
        logger.info(f"Получен новый запрос к чат-эндпоинту")
        logger.info(f"Содержание вопроса: {request.question}")
        
        logger.debug("Получение графа RAG из состояния приложения")
        graph = fastapi_request.app.state.rag_graph
        
        logger.debug("Формирование начального состояния сообщения")
        message_state = {"messages": [{"role": "user", "content": request.question}], "context": ""}
        logger.debug(f"Сформировано состояние сообщения: {message_state}")

        logger.debug("Настройка конфигурации для запроса к LLM")
        config = RunnableConfig(
            configurable={
                "api_key": settings.OPENAI_API_KEY,
                "proxy": settings.OPENAI_PROXY,
                "tokens": settings.GENERATION_TOKENS,
                "temperature": settings.GENERATION_TEMPERATURE,
                "model_name_gpt": settings.MODEL_NAME,
                "model_name_embedding": settings.MODEL_EMBEDDING_NAME,
                "vectorstore": fastapi_request.app.state.database
            }
        )
        logger.debug(f"Конфигурация настроена с моделью GPT: {settings.MODEL_NAME}")

        logger.info("Отправка запроса к графу RAG")
        ai_response = await graph.ainvoke(message_state, config=config)
        logger.info("Получен ответ от графа RAG")
        logger.debug(f"Содержание ответа: {ai_response['messages'].content}")

        logger.info("Формирование ответа для клиента")
        return ChatResponse(
            answer=ai_response["messages"].content
        )
    except Exception as e:
        logger.error(f"Произошла ошибка при обработке запроса: {str(e)}")
        logger.exception("Полный стек ошибки:")
        raise HTTPException(status_code=500, detail=str(e))

