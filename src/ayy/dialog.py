import json
from copy import deepcopy
from datetime import datetime
from enum import StrEnum
from functools import partial
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Annotated, Any, Literal

import instructor
from anthropic import Anthropic, AsyncAnthropic
from google.generativeai import GenerativeModel
from loguru import logger
from openai import AsyncOpenAI, OpenAI
from pydantic import AfterValidator, BaseModel
from sqlmodel import Field, Relationship, SQLModel


class ModelName(StrEnum):
    GPT = "gpt-4o-2024-08-06"
    GPT_MINI = "gpt-4o-mini"
    HAIKU = "claude-3-haiku-20240307st"
    SONNET = "claude-3-5-sonnet-latest"
    OPUS = "claude-3-opus-latest"
    GEMINI_PRO = "gemini-1.5-pro-001"
    GEMINI_FLASH = "gemini-1.5-flash-002"
    GEMINI_FLASH_EXP = "gemini-1.5-flash-exp-0827"


MessageType = dict[str, Any]

TRIMMED_LEN = 40
MERGE_JOINER = "\n\n--- Next Message ---\n\n"
MODEL_NAME = ModelName.GEMINI_FLASH
TEMPERATURE = 0.1
MAX_TOKENS = 3000


def create_creator(
    model_name: ModelName = MODEL_NAME, use_async: bool = False
) -> instructor.Instructor | instructor.AsyncInstructor:
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


def load_content(content: Any, echo: bool = True) -> Any:
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


def chat_message(role: str, content: Content, template: Content = "") -> MessageType:
    if template:
        if not isinstance(content, dict):
            raise TypeError("When using template, content must be a dict.")
        try:
            message_content = template.format(**content)
        except KeyError as e:
            raise KeyError(f"Template {template} requires key {e} which was not found in content.")
    else:
        message_content = content
    return {"role": role, "content": message_content}


def system_message(content: Content, template: Content = "") -> MessageType:
    return chat_message(role="system", content=content, template=template)


def user_message(content: Content, template: Content = "") -> MessageType:
    return chat_message(role="user", content=content, template=template)


def assistant_message(content: Content, template: Content = "") -> MessageType:
    return chat_message(role="assistant", content=content, template=template)


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


def trim_messages(messages: Messages, trimmed_len: int = TRIMMED_LEN) -> list[MessageType]:
    if len(messages) <= trimmed_len:
        return messages
    for start_idx in range(len(messages) - trimmed_len, -1, -1):
        trimmed_messages = messages[start_idx:]
        if trimmed_messages[0]["role"] == "user":
            if messages[0]["role"] == "system":
                trimmed_messages.insert(0, messages[0])
            return trimmed_messages
    return messages


def messages_to_kwargs(
    messages: Messages, system: str = "", model_name: str = MODEL_NAME, joiner: Content = MERGE_JOINER
) -> dict:
    messages = deepcopy(messages)
    kwargs = {"messages": messages}
    first_message = messages[0]
    if first_message["role"] == "system":
        system = system or first_message["content"]
        kwargs["messages"][0]["content"] = system
    else:
        kwargs["messages"].insert(0, system_message(content=system))
    if any(name in model_name.lower() for name in ("gemini", "claude")):
        kwargs["messages"] = merge_same_role_messages(messages=kwargs["messages"], joiner=joiner)
    if "claude" in model_name.lower():
        return {"system": system, "messages": kwargs["messages"][1:]}
    return kwargs


class SQL_Dialog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    system: str = ""
    model_name: str = MODEL_NAME
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    messages: list["SQL_Message"] = Relationship(back_populates="sql_dialog")


class SQL_Message(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    dialog_id: int = Field(foreign_key="sql_dialog.id")
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    sql_dialog: SQL_Dialog = Relationship(back_populates="messages")


class Dialog(BaseModel):
    system: Content = ""
    messages: Messages = Field(default_factory=list)
    model_name: str = MODEL_NAME
    creation_config: dict = dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    memory_tags: list[Literal["core", "recall"]] = Field(default_factory=list)

    def to_sql_dialog(self) -> SQL_Dialog:
        return SQL_Dialog(
            system=self.system, model_name=self.model_name, messages=[SQL_Message(**msg) for msg in self.messages]
        )


def dialog_to_kwargs(dialog: Dialog) -> dict:
    kwargs = messages_to_kwargs(messages=dialog.messages, system=dialog.system, model_name=dialog.model_name)
    if "gemini" in dialog.model_name.lower():
        kwargs["generation_config"] = dialog.creation_config
    else:
        kwargs.update(dialog.creation_config)
    return kwargs


def add_assistant_message(dialog: Dialog, creator: Content) -> Dialog:
    try:
        res = (
            creator.create(
                **messages_to_kwargs(
                    messages=deepcopy(dialog.messages), system=dialog.system, model_name=dialog.model_name
                )
            )
            if isinstance(creator, (instructor.Instructor, instructor.AsyncInstructor))
            else creator
        )
    except Exception as e:
        logger.exception(f"Error in respond. Last message: {dialog.messages[-1]}")
        res = f"Error: {e}"
    dialog.messages.append(assistant_message(content=res))
    return dialog


def add_user_message(dialog: Dialog, content: Content, template: Content = "") -> Dialog:
    dialog.messages.append(user_message(content=content, template=template))
    return dialog


print("hello")
