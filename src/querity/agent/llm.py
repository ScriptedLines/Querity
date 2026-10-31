from __future__ import annotations

import json
import random
import re
import time
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from querity.agent.settings import AgentSettings
from querity.ingestion.llm import build_chat_model

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    pass


def text_from_message(message: Any) -> str:
    if isinstance(message, str):
        return message
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                value = block.get("text") or block.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    return str(content)


def _parse_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError("Model response did not contain a JSON object")


def invoke_structured[T: BaseModel](
    llm: Any,
    schema: type[T],
    messages: list[BaseMessage],
    *,
    max_retries: int = 3,
) -> T:
    structured: Any | None = None
    try:
        structured = llm.with_structured_output(schema)
    except Exception:
        structured = None

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if structured is not None:
                result = structured.invoke(messages)
            else:
                result = llm.invoke(
                    [
                        *messages,
                        SystemMessage(
                            content=(
                                "Return only a JSON object matching this schema, with no markdown "
                                f"or commentary:\n{json.dumps(schema.model_json_schema())}"
                            )
                        ),
                    ]
                )
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
            if isinstance(result, AIMessage):
                return schema.model_validate(_parse_json(text_from_message(result)))
            return schema.model_validate(_parse_json(text_from_message(result)))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
        if attempt < max_retries:
            time.sleep(min(2**attempt + random.uniform(0, 0.5), 10))
    raise StructuredOutputError(
        f"Could not obtain valid {schema.__name__} from the model: {last_error}"
    )


class AgentModels:
    def __init__(self, settings: AgentSettings) -> None:
        self._settings = settings
        self.planner = build_chat_model(settings.model_copy(update={"llm_temperature": 0.0}))
        self.generator = build_chat_model(settings)
        self.reviewer = build_chat_model(
            settings.model_copy(
                update={
                    "llm_model": settings.effective_review_model,
                    "llm_temperature": 0.0,
                }
            )
        )

    @property
    def retries(self) -> int:
        return self._settings.llm_max_retries

    def plan(self, messages: list[BaseMessage], schema: type[T]) -> T:
        return invoke_structured(self.planner, schema, messages, max_retries=self.retries)

    def generate_sql(self, messages: list[BaseMessage], schema: type[T]) -> T:
        return invoke_structured(self.generator, schema, messages, max_retries=self.retries)

    def review(self, messages: list[BaseMessage], schema: type[T]) -> T:
        return invoke_structured(self.reviewer, schema, messages, max_retries=self.retries)


def build_messages(system: str, human: str) -> list[BaseMessage]:
    return [SystemMessage(content=system), HumanMessage(content=human)]
