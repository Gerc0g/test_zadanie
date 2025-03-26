from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_community.vectorstores import FAISS

@dataclass(kw_only=True)
class Configuration:
    tokens: Optional[int] = None
    temperature: Optional[float] = None
    api_key: str
    proxy: str
    model_name_gpt: str
    model_name_embedding: str
    vectorstore: 'FAISS'

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        configurable = (config.get("configurable") or {}) if config else {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})