from enum import StrEnum
from functools import partial
from itertools import groupby
from operator import itemgetter
from typing import Any

MessageType = dict[str, Any]


class ModelName(StrEnum):
    GPT = "gpt-4o-2024-08-06"
    GPT_MINI = "gpt-4o-mini"
    HAIKU = "claude-3-haiku-20240307"
    SONNET = "claude-3-5-sonnet-20240620"
    OPUS = "claude-3-opus-20240229"
    GEMINI_PRO = "gemini-1.5-pro-001"
    GEMINI_FLASH = "gemini-1.5-flash-002"
    GEMINI_FLASH_EXP = "gemini-1.5-flash-exp-0827"


TRIMMED_LEN = 40
MERGE_JOINER = "\n\n--- Next Message ---\n\n"
MODEL = ModelName.GEMINI_FLASH


def chat_message(role: str, content: Any, template: str = "") -> MessageType:
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


def system_message(content: Any, template: str = "") -> MessageType:
    return chat_message(role="system", content=content, template=template)


def user_message(content: Any, template: str = "") -> MessageType:
    return chat_message(role="user", content=content, template=template)


def assistant_message(content: Any, template: str = "") -> MessageType:
    return chat_message(role="assistant", content=content, template=template)


def exchange(
    user: Any,
    assistant: Any,
    feedback: Any = "",
    correction: Any = "",
    user_template: str = "",
    assistant_template: str = "",
) -> list[MessageType]:
    user_maker = partial(user_message, template=user_template)
    assistant_maker = partial(assistant_message, template=assistant_template)
    return (
        [user_maker(content=user), assistant_maker(content=assistant)]
        + ([user_maker(content=feedback)] if feedback else [])
        + ([assistant_maker(content=correction)] if correction else [])
    )


def merge_same_role_messages(messages: list[MessageType], joiner: str = MERGE_JOINER) -> list[MessageType]:
    return (
        [
            {"role": role, "content": joiner.join(msg["content"] for msg in group)}
            for role, group in groupby(messages, key=itemgetter("role"))
        ]
        if messages
        else []
    )


def trim_messages(messages: list[MessageType], trimmed_len: int = TRIMMED_LEN) -> list[MessageType]:
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
    messages: list[MessageType], model_name: ModelName = MODEL, defualt_task: str = "", joiner: str = MERGE_JOINER
) -> dict[str, str | list[MessageType]]:
    kwargs = {
        "system": messages[0]["content"] if messages[0]["role"] == "system" else defualt_task,
        "messages": messages,
    }
    if "gpt" not in model_name.lower():
        kwargs["messages"] = merge_same_role_messages(messages=messages, joiner=joiner)
    return kwargs
