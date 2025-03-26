from functools import lru_cache

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

from .src.configuration import Configuration
from .src.state import AgentState
from dotenv import load_dotenv
from config import settings
from utils.logger import logger

load_dotenv()


@lru_cache(maxsize=5)
def _get_model(temperature: float, api_key: str, proxy: str, tokens: int = 16384, model_name: str = "gpt-4o"):
    logger.debug(f"Создание модели ChatOpenAI с параметрами: temperature={temperature}, model_name={model_name}")
    
    model_params = {
        "temperature": temperature,
        "model_name": model_name,
        "openai_api_key": api_key,
        "max_tokens": tokens
    }
    
    if proxy:
        model_params["openai_proxy"] = proxy
        
    model = ChatOpenAI(**model_params)
    
    logger.debug("Модель ChatOpenAI успешно создана")
    return model


async def retrieve(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.info("Начало процесса извлечения контекста")
    configuration = Configuration.from_runnable_config(config)
    query = state["messages"][-1]["content"]
    logger.debug(f"Получен запрос для поиска: {query}")
    
    logger.debug(f"Загрузка векторного хранилища из {configuration.vectorstore}")

    logger.debug("Инициализация модели для MultiQueryRetriever")
    llm = _get_model(temperature=0,
                 api_key=configuration.api_key,
                 proxy=configuration.proxy,
                 model_name=configuration.model_name_gpt)
    
    logger.debug("Создание MultiQueryRetriever")
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=configuration.vectorstore.as_retriever(search_kwargs = {"k" : 10, 'filter': {'table':'chunks_1200_600'}}), llm=llm
    )
    
    logger.debug("Поиск релевантных документов")
    relevant_docs = retriever_from_llm.invoke(query)
    logger.debug(f"Найдено {len(relevant_docs)} релевантных документов")

    context = "\n".join([doc.page_content for doc in relevant_docs])
    logger.debug("Контекст успешно собран")
    
    logger.info("Процесс извлечения контекста завершен")
    return {"context": context, "messages": state["messages"]}


async def generate(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.info("Начало генерации ответа")
    configuration = Configuration.from_runnable_config(config)
    messages = state["messages"]
    context = state["context"]

    logger.debug("Инициализация модели для генерации ответа")
    model = _get_model(
        temperature=configuration.temperature,
        api_key=configuration.api_key,
        proxy=configuration.proxy,
        tokens=configuration.tokens,
        model_name=configuration.model_name_gpt
    )

    system_prompt = """Ты - помощник, который отвечает на вопросы, используя только предоставленный контекст.
    Если в контексте нет информации для ответа, скажи об этом.
    Подробно рассписывай ответ, используя контекст, это необходимо нашим пользователям чтобы они могли понять ответ.
    Не используй в ответе специальные символы форматирования как *, _, #.
    Если в контексте нет информации для ответа, скажи об этом и постарайся кратко ответить из своих знаний."""
    
    logger.debug("Форматирование сообщений для модели")
    formatted_messages = [
        {"role": "system", "content": f"{system_prompt}\nКонтекст: {context}"}
    ] + [{"role": m["role"], "content": m["content"]} for m in messages]

    logger.debug("Отправка запроса к модели")
    response = await model.ainvoke(formatted_messages)
    logger.debug("Ответ от модели получен")
    
    logger.info("Генерация ответа завершена")
    return {"messages": response}


logger.info("Инициализация графа RAG")
workflow = StateGraph(AgentState, config_schema=Configuration)

logger.debug("Добавление узлов в граф")
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

logger.debug("Добавление связей между узлами")
workflow.add_edge("__start__", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "__end__")

logger.info("Компиляция графа RAG")
rag_graph = workflow.compile()
