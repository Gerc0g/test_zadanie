from functools import lru_cache

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from langgraph_app.src.configuration import Configuration
from langgraph_app.src.state import RagState
from langchain_openai import ChatOpenAI



@lru_cache(maxsize=1)
def _get_model(temperature: float, api_key: str, proxy: str, tokens: int = 16384):
    model = ChatOpenAI(
            temperature=temperature,
            model_name="gpt-4o",
            openai_api_key=api_key,
            openai_proxy=proxy,
            max_tokens=tokens
    )
    return model

async def ai_response(state: RagState, config: RunnableConfig):

    configuration = Configuration.from_runnable_config(config)
    messages = state["messages"]
    context = state["context"]
    system_prompt = state["system_prompt"]

    model = _get_model(
        temperature=configuration.temperature,
        api_key=configuration.api_key,
        proxy=configuration.proxy,
        tokens=configuration.tokens
    )

    messages = [{"role": "system", "content": f"{system_prompt}. \n Контекст проекта: {context}"}] + messages

    response = await model.ainvoke(messages)
    return {"messages": response}



workflow = StateGraph(RagState, config_schema=Configuration)

workflow.add_node("ai_response", ai_response)

workflow.add_edge("__start__", "ai_response")
workflow.add_edge("ai_response", "__end__")


rag_graph = workflow.compile()
