import json
from copy import deepcopy
from enum import StrEnum
from functools import partial
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Annotated, Any, Self
from uuid import uuid4

import instructor
import numpy as np
from anthropic import Anthropic, AsyncAnthropic
from google.generativeai import GenerativeModel
from instructor import AsyncInstructor, Instructor
from loguru import logger
from openai import AsyncOpenAI, OpenAI
from pydantic import UUID4, AfterValidator, BaseModel, Field, field_validator, model_validator

from ayy.memory import Summary
from ayy.prompts import SELECT_TOOLS
from ayy.utils import flatten_list

MAX_MESSAGE_TOKENS = 75_000
MERGE_JOINER = "\n\n--- Next Message ---\n\n"
TEMPERATURE = 0.1
MAX_TOKENS = 3000


class ModelName(StrEnum):
    GPT = "gpt-4o-2024-11-20"
    GPT_MINI = "gpt-4o-mini"
    HAIKU = "claude-3-haiku-latest"
    SONNET = "claude-3-5-sonnet-latest"
    OPUS = "claude-3-opus-latest"
    GEMINI_PRO = "gemini-1.5-pro-001"
    GEMINI_FLASH = "gemini-1.5-flash-002"
    GEMINI_FLASH_EXP = "gemini-2.0-flash-exp"


MODEL_NAME = ModelName.GEMINI_FLASH


def load_content(content: Any, echo: bool = False) -> Any:
    if not isinstance(content, (str, Path)):
        return content
    else:
        try:
            return Path(content).read_text()
        except Exception as e:
            if echo:
                logger.warning(f"Could not load content as a file: {str(e)[:100]}")
            return str(content)


Content = Annotated[Any, AfterValidator(load_content)]
MessageType = dict[str, Content]


def create_creator(model_name: ModelName = MODEL_NAME, use_async: bool = False) -> Instructor | AsyncInstructor:
    if "gpt" in model_name.lower():
        if use_async:
            client = instructor.from_openai(AsyncOpenAI())
        else:
            client = instructor.from_openai(OpenAI())
    elif "claude" in model_name.lower():
        if use_async:
            client = instructor.from_anthropic(AsyncAnthropic())
        else:
            client = instructor.from_anthropic(Anthropic())
    elif "gemini" in model_name.lower():
        client = instructor.from_gemini(
            client=GenerativeModel(model_name=model_name),
            mode=instructor.Mode.GEMINI_JSON,
            use_async=use_async,  # type: ignore
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    return client.chat.completions


def chat_message(role: str, content: Content, template: Content = "", **kwargs) -> MessageType:
    if template:
        if not isinstance(content, dict):
            raise TypeError("When using template, content must be a dict.")
        try:
            message_content = template.format(**content)
        except KeyError as e:
            raise KeyError(f"Template {template} requires key {e} which was not found in content.")
    else:
        message_content = content
    return {"role": role, "content": message_content, **kwargs}


def system_message(content: Content, template: Content = "", **kwargs) -> MessageType:
    return chat_message(role="system", content=content, template=template, **kwargs)


def user_message(content: Content, template: Content = "", **kwargs) -> MessageType:
    return chat_message(role="user", content=content, template=template, **kwargs)


def assistant_message(content: Content, template: Content = "", **kwargs) -> MessageType:
    return chat_message(role="assistant", content=content, template=template, **kwargs)


def load_messages(messages: list[MessageType] | str | Path) -> list[MessageType]:
    if isinstance(messages, list):
        return messages
    else:
        try:
            return json.loads(Path(messages).read_text())
        except Exception as e:
            logger.warning(f"Could not load messages as a file: {str(e)[:100]}")
            return [user_message(content=str(messages))]


Messages = Annotated[list[MessageType], AfterValidator(load_messages)]


class AgentToolSignature(BaseModel):
    name: str
    signature: str
    docstring: str
    system: str

    def to_dict(self) -> dict:
        return {"name": self.name, "signature": self.signature, "docstring": self.docstring}


class Agent(BaseModel):
    id: UUID4 = Field(default_factory=lambda: uuid4())
    name: str = ""
    system: Content = ""
    messages: Messages = Field(default_factory=list)
    model_name: ModelName = MODEL_NAME
    creation_config: dict = dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    max_message_tokens: int | None = None
    available_tools: list[str] = Field(default_factory=list)
    include_tool_guidelines: bool = True
    summarized_tasks: list[str] = Field(default_factory=list)
    agent_tool_signature: AgentToolSignature | None = None

    @model_validator(mode="after")
    def validate_signature(self) -> Self:
        if self.agent_tool_signature is None:
            return self
        self.name = self.agent_tool_signature.name.strip() or self.name
        self.system = self.agent_tool_signature.system.strip() or self.system
        return self

    @model_validator(mode="after")
    def validate_system(self) -> Self:
        if self.include_tool_guidelines:
            self.system += f"\n\n<tool_selection_guidelines>\n{SELECT_TOOLS}\n</tool_selection_guidelines>"
        self.system = self.system.strip()
        return self

    @field_validator("creation_config")
    @classmethod
    def validate_creation_config(cls, v: dict) -> dict:
        return {
            **v,
            "temperature": v.get("temperature", TEMPERATURE),
            "max_tokens": v.get("max_tokens", MAX_TOKENS),
        }

    @model_validator(mode="after")
    def validate_max_message_tokens(self) -> Self:
        if self.max_message_tokens is not None:
            return self
        if "gpt" in self.model_name.lower():
            self.max_message_tokens = 75_000
        elif "claude" in self.model_name.lower():
            self.max_message_tokens = 100_000
        elif "gemini" in self.model_name.lower():
            self.max_message_tokens = 250_000
        return self

    @field_validator("summarized_tasks")
    @classmethod
    def validate_summarized_tasks(cls, v: list[str]) -> list[str]:
        return sorted(set(v))


def get_last_message(messages: Messages, role: str = "assistant") -> MessageType | None:
    return next((msg for msg in reversed(messages) if msg["role"] == role), None)


def exchange(
    user: Content,
    assistant: Content,
    feedback: Content = "",
    correction: Content = "",
    user_template: Content = "",
    assistant_template: Content = "",
) -> list[MessageType]:
    user_maker = partial(user_message, template=user_template)
    assistant_maker = partial(assistant_message, template=assistant_template)
    return (
        [user_maker(content=user), assistant_maker(content=assistant)]
        + ([user_maker(content=feedback)] if feedback else [])
        + ([assistant_maker(content=correction)] if correction else [])
    )


def merge_same_role_messages(messages: Messages, joiner: Content = MERGE_JOINER) -> list[MessageType]:
    return (
        [
            {"role": role, "content": joiner.join(msg["content"] for msg in group)}
            for role, group in groupby(messages, key=itemgetter("role"))
        ]
        if messages
        else []
    )


def count_tokens(messages: Messages) -> int:
    return int(np.ceil(len(flatten_list([message["content"].split() for message in messages])) * 0.75))


def get_trim_index(messages: Messages, max_message_tokens: int | None = MAX_MESSAGE_TOKENS) -> int:
    if max_message_tokens is None or len(messages) == 0:
        return 0
    total_tokens = 0
    trim_idx = len(messages) - 1
    while trim_idx >= 0:
        msg_tokens = count_tokens([messages[trim_idx]])
        if total_tokens + msg_tokens > max_message_tokens:
            break
        total_tokens += msg_tokens
        trim_idx -= 1
    return trim_idx + 1


def messages_to_kwargs(
    messages: Messages, system: str = "", model_name: str = MODEL_NAME, joiner: Content = MERGE_JOINER
) -> dict:
    messages = deepcopy(messages)
    kwargs = {"messages": [chat_message(role=message["role"], content=message["content"]) for message in messages]}
    if messages and messages[0]["role"] == "system":
        system = system or messages[0]["content"]
        kwargs["messages"][0]["content"] = system
    else:
        kwargs["messages"].insert(0, system_message(content=system))
    if len(kwargs["messages"]) > 1 and kwargs["messages"][1]["role"] == "assistant":
        kwargs["messages"][1]["role"] = "user"
        kwargs["messages"][1]["content"] = (
            f"<previous_assistant_message>\n{kwargs['messages'][1]['content']}\n</previous_assistant_message>"
        )
    if any(name in model_name.lower() for name in ("gemini", "claude")):
        kwargs["messages"] = merge_same_role_messages(messages=kwargs["messages"], joiner=joiner)
    if "claude" in model_name.lower():
        return {"system": system, "messages": kwargs["messages"][1:]}
    return kwargs


def agent_to_kwargs(agent: Agent, messages: Messages | None = None, joiner: Content = MERGE_JOINER) -> dict:
    messages = messages or []
    kwargs = messages_to_kwargs(
        messages=agent.messages + [msg for msg in messages if msg["role"] != "system"],
        system=agent.system,
        model_name=agent.model_name,
        joiner=joiner,
    )
    if "gemini" in agent.model_name.lower():
        kwargs["generation_config"] = agent.creation_config
    else:
        kwargs.update(agent.creation_config)
    logger.warning(f"kwargs: {kwargs}")
    return kwargs


def summarize_messages(
    messages: Messages, summarizer_agent: Agent | None = None, joiner: Content = MERGE_JOINER
) -> Summary | None:
    if not messages or summarizer_agent is None:
        return None
    logger.info(f"Summarizing messages: {messages[-2:]}")
    try:
        creator = create_creator(model_name=summarizer_agent.model_name)
        summary: Summary = creator.create(
            **agent_to_kwargs(
                agent=summarizer_agent,
                messages=messages + [user_message(content="Summarize the conversation so far.")],
                joiner=joiner,
            ),
            response_model=Summary,  # type: ignore
        )
        logger.success("Summarized messages")
        return summary
    except Exception:
        logger.exception(f"Could not summarize messages: {messages[-2:]}")
        return None
