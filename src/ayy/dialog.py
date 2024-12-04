import json
from copy import deepcopy
from enum import Enum, StrEnum
from functools import partial
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

import instructor
from anthropic import Anthropic, AsyncAnthropic
from google.generativeai import GenerativeModel
from instructor import AsyncInstructor, Instructor
from loguru import logger
from openai import AsyncOpenAI, OpenAI
from pydantic import UUID4, AfterValidator, BaseModel, Field, field_validator

from ayy.prompts import TOOL_SELECTION_SYSTEM

TRIMMED_LEN = 40
MERGE_JOINER = "\n\n--- Next Message ---\n\n"
TEMPERATURE = 0.1
MAX_TOKENS = 3000
DEFAULT_TAG = "RECALL"


class Tool(BaseModel):
    id: UUID4 | str = Field(default_factory=lambda: str(uuid4()))
    reasoning: str
    name: str
    prompt: str = Field(
        ...,
        description="An LLM will receive the messages so far and the tools calls and results up until now. This prompt will then be used to ask the LLM to generate arguments for the selected tool based on the tool's signature. If the tool doesn't have any parameters, then it doesn't need a prompt.",
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: UUID4 | str) -> str:
        if isinstance(v, str):
            v = uuid4()
        return str(v)

    def __str__(self) -> str:
        return f"Reasoning: {self.reasoning}\nName: {self.name}\nPrompt: {self.prompt}"


class ModelName(StrEnum):
    GPT = "gpt-4o-2024-11-20"
    GPT_MINI = "gpt-4o-mini"
    HAIKU = "claude-3-haiku-latest"
    SONNET = "claude-3-5-sonnet-latest"
    OPUS = "claude-3-opus-latest"
    GEMINI_PRO = "gemini-1.5-pro-001"
    GEMINI_FLASH = "gemini-1.5-flash-002"
    GEMINI_FLASH_EXP = "gemini-1.5-flash-exp-0827"


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


MODEL_NAME = ModelName.GEMINI_FLASH
Content = Annotated[Any, AfterValidator(load_content)]
MessageType = dict[str, Content]


class MessagePurpose(StrEnum):
    CONVO = "convo"
    TOOL = "tool"


DEFAULT_MESSAGE_PURPOSE = MessagePurpose.CONVO


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


def chat_message(
    role: str, content: Content, template: Content = "", purpose: MessagePurpose = DEFAULT_MESSAGE_PURPOSE
) -> MessageType:
    if template:
        if not isinstance(content, dict):
            raise TypeError("When using template, content must be a dict.")
        try:
            message_content = template.format(**content)
        except KeyError as e:
            raise KeyError(f"Template {template} requires key {e} which was not found in content.")
    else:
        message_content = content
    return {"role": role, "content": message_content, "purpose": purpose}


def system_message(
    content: Content, template: Content = "", purpose: MessagePurpose = DEFAULT_MESSAGE_PURPOSE
) -> MessageType:
    return chat_message(role="system", content=content, template=template, purpose=purpose)


def user_message(
    content: Content, template: Content = "", purpose: MessagePurpose = DEFAULT_MESSAGE_PURPOSE
) -> MessageType:
    return chat_message(role="user", content=content, template=template, purpose=purpose)


def assistant_message(
    content: Content, template: Content = "", purpose: MessagePurpose = DEFAULT_MESSAGE_PURPOSE
) -> MessageType:
    return chat_message(role="assistant", content=content, template=template, purpose=purpose)


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
    messages: Messages,
    system: str = "",
    model_name: str = MODEL_NAME,
    joiner: Content = MERGE_JOINER,
    trimmed_len: int = TRIMMED_LEN,
) -> dict:
    messages = deepcopy(messages)
    messages = trim_messages(messages=messages, trimmed_len=trimmed_len)
    kwargs = {"messages": [chat_message(role=message["role"], content=message["content"]) for message in messages]}
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


class DialogToolSignature(BaseModel):
    name: str = ""
    signature: str = ""
    docstring: str = ""


class Dialog(BaseModel):
    id: UUID4 = Field(default_factory=lambda: uuid4())
    system: Content = ""
    messages: Messages = Field(default_factory=list)
    model_name: ModelName = MODEL_NAME
    creation_config: dict = dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    name: str = ""
    dialog_tool_signature: dict = Field(default_factory=dict)
    available_tools: list[str] = Field(default_factory=list)

    @field_validator("system")
    @classmethod
    def validate_system(cls, v: Content) -> Content:
        return (
            v.strip() + f"\n\n<tool_selection_guidelines>\n{TOOL_SELECTION_SYSTEM}\n</tool_selection_guidelines>"
        )


class DialogTool(BaseModel):
    id: int
    position: int
    dialog_id: UUID4
    tool: Tool
    used: bool = False


def dialog_to_kwargs(dialog: Dialog, messages: Messages | None = None, trimmed_len: int = TRIMMED_LEN) -> dict:
    kwargs = messages_to_kwargs(
        messages=messages or dialog.messages,
        system=dialog.system,
        model_name=dialog.model_name,
        trimmed_len=trimmed_len,
    )
    if "gemini" in dialog.model_name.lower():
        kwargs["generation_config"] = dialog.creation_config
    else:
        kwargs.update(dialog.creation_config)
    # logger.info(f"KWARGS: {kwargs}")
    return kwargs


class MemoryTagInfo(BaseModel):
    description: str
    use_cases: list[str]


class MemoryTag(Enum):
    CORE = MemoryTagInfo(
        description="Messages that are crucial and should persist even after the dialog concludes",
        use_cases=[
            "Factual information that won't change",
            "Important facts about the user",
            "Long-term preferences",
            "Critical instructions or rules",
            "User feedback intended to improve future interactions",
        ],
    )
    RECALL = MemoryTagInfo(
        description="Messages relevant to the current task and should be remembered during the ongoing dialog",
        use_cases=["Current task parameters", "Intermediate results", "Temporary user preferences"],
    )


class MemoryTags(BaseModel):
    reasoning: str
    memory_tags: list[Literal[*MemoryTag._member_names_]]  # type: ignore

    @field_validator("memory_tags")
    @classmethod
    def validate_memory_tags(cls, v: list[str]) -> list[str]:
        return list(set(v) | {DEFAULT_TAG})


def add_message(
    dialog: Dialog,
    role: str = "user",
    content: Content = "",
    template: Content = "",
    purpose: MessagePurpose = DEFAULT_MESSAGE_PURPOSE,
    message: MessageType | None = None,
    memory_tagger_dialog: Dialog | None = None,
    tagger_trimmed_len: int = TRIMMED_LEN,
) -> Dialog:
    if content is None:
        return dialog
    message = message or chat_message(role=role, content=content, template=template, purpose=purpose)
    if memory_tagger_dialog is not None:
        try:
            memory_tagger_dialog.messages += dialog.messages + [message]
            creator = create_creator(model_name=memory_tagger_dialog.model_name)
            memory_tags: MemoryTags = creator.create(
                **dialog_to_kwargs(dialog=memory_tagger_dialog, trimmed_len=tagger_trimmed_len),
                response_model=MemoryTags,  # type: ignore
            )
        except Exception:
            logger.exception(
                f"Could not add tag to message: {message['content'][:100]}\nDefaulting to {DEFAULT_TAG}"
            )
            memory_tags = MemoryTags(reasoning="", memory_tags=[DEFAULT_TAG])
        message["memory_tags"] = memory_tags.memory_tags
    dialog.messages.append(message)
    return dialog
