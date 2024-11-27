import json
from copy import deepcopy
from enum import Enum, StrEnum
from functools import partial
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Annotated, Any, Literal

import instructor
from anthropic import Anthropic, AsyncAnthropic
from google.generativeai import GenerativeModel
from instructor import AsyncInstructor, Instructor
from loguru import logger
from openai import AsyncOpenAI, OpenAI
from pydantic import AfterValidator, BaseModel, Field

TRIMMED_LEN = 40
MERGE_JOINER = "\n\n--- Next Message ---\n\n"
TEMPERATURE = 0.1
MAX_TOKENS = 3000
DEFAULT_PROMPT = "Generate a response if you've been asked. Otherwise, ask the user how they are doing."
DEFAULT_TAG = "RECALL"


class ModelName(StrEnum):
    GPT = "gpt-4o-2024-08-06"
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


class Dialog(BaseModel):
    system: Content = ""
    messages: Messages = Field(default_factory=list)
    model_name: ModelName = MODEL_NAME
    creation_config: dict = dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)


def dialog_to_kwargs(dialog: Dialog, trimmed_len: int = TRIMMED_LEN) -> dict:
    kwargs = messages_to_kwargs(
        messages=dialog.messages, system=dialog.system, model_name=dialog.model_name, trimmed_len=trimmed_len
    )
    if "gemini" in dialog.model_name.lower():
        kwargs["generation_config"] = dialog.creation_config
    else:
        kwargs.update(dialog.creation_config)
    logger.info(f"kwargs: {kwargs}")
    return kwargs


class MemoryTagInfo(BaseModel):
    description: str
    use_cases: list[str]


class MemoryTag(Enum):
    CORE = MemoryTagInfo(
        description="Messages that are crucial and should persist even after the dialog concludes",
        use_cases=[
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


def add_message(
    dialog: Dialog,
    role: str = "user",
    content: Content = "",
    template: Content = "",
    message: MessageType | None = None,
    tagger_dialog: Dialog | None = None,
    tagger_trimmed_len: int = TRIMMED_LEN,
) -> Dialog:
    if not message:
        message = chat_message(role=role, content=content, template=template)
    if tagger_dialog is not None:
        try:
            tagger_dialog.system = (
                tagger_dialog.system or f"Tag the latest message.Possible tags are {str(MemoryTag.__members__)}"
            )
            tagger_dialog.messages += dialog.messages + [message]
            creator = create_creator(model_name=tagger_dialog.model_name)
            tags: list[str] = creator.create(
                **dialog_to_kwargs(dialog=tagger_dialog, trimmed_len=tagger_trimmed_len),
                response_model=list[Literal[*MemoryTag._member_names_]],  # type: ignore
            )
        except Exception:
            logger.exception(
                f"Could not add tag to message: {message['content'][:100]}\nDefaulting to {DEFAULT_TAG}"
            )
            tags = [DEFAULT_TAG]
        message["tags"] = list(set(tags) | {DEFAULT_TAG})
    dialog.messages.append(message)
    return dialog
