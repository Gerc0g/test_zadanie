from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from utils.logger import logger

from routes import chat
from langgraph_app.chat_graph import rag_graph
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск приложения...")
    logger.info("Инициализация графа RAG...")
    app.state.rag_graph = rag_graph
    logger.info("Граф RAG успешно инициализирован")

    logger.debug("Инициализация embeddings с моделью %s", settings.MODEL_EMBEDDING_NAME)
    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_proxy=settings.OPENAI_PROXY,
        model=settings.MODEL_EMBEDDING_NAME
    )
    logger.debug("Embeddings успешно созданы")
    
    logger.debug(f"Загрузка векторного хранилища из  {settings.PATH_TO_VECTOR_STORE}")
    app.state.database = FAISS.load_local(settings.PATH_TO_VECTOR_STORE, embeddings, allow_dangerous_deserialization=True)
    logger.debug("Векторное хранилище успешно загружено")
    logger.debug(f"Тип хранилища: {type(app.state.database)}")
    logger.debug(f"Содержимое векторного хранилища: {len(app.state.database.docstore._dict)}")
    logger.info("Приложение готово к работе")
    yield

    logger.info("Начало процесса завершения работы...")
    logger.debug("Очистка состояния приложения...")
    logger.debug("Удаление графа RAG из состояния...")
    del app.state.rag_graph
    logger.debug("Удаление векторного хранилища из состояния...")
    del app.state.database
    logger.info("Граф RAG и база данных успешно удалены")
    logger.info("Приложение успешно завершило работу")

app = FastAPI(lifespan=lifespan)

logger.info("Настройка CORS middleware...")
logger.debug("Установка параметров CORS...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware успешно настроен")

logger.info("Подключение маршрутов чата...")
app.include_router(chat.router)
logger.info("Маршруты чата успешно подключены")

@app.get("/")
def read_root():
    logger.debug("Получен GET запрос к корневому эндпоинту '/'")
    logger.info("Отправка ответа с приветственным сообщением")
    return {"message": "Тестовое задание RAG"}