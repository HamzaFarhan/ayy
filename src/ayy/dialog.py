import json
from copy import deepcopy
from enum import StrEnum
from functools import partial
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Annotated, Any

import instructor
from burr.core import ApplicationBuilder, action
from burr.integrations.pydantic import PydanticTypingSystem
from burr.tracking import LocalTrackingClient
from google.generativeai import GenerativeModel
from loguru import logger
from pydantic import AfterValidator, BaseModel, Field


class ModelName(StrEnum):
    GPT = "gpt-4o-2024-08-06"
    GPT_MINI = "gpt-4o-mini"
    HAIKU = "claude-3-haiku-20240307"
    SONNET = "claude-3-5-sonnet-20240620"
    OPUS = "claude-3-opus-20240229"
    GEMINI_PRO = "gemini-1.5-pro-001"
    GEMINI_FLASH = "gemini-1.5-flash-002"
    GEMINI_FLASH_EXP = "gemini-1.5-flash-exp-0827"


MessageType = dict[str, Any]

TRIMMED_LEN = 40
MERGE_JOINER = "\n\n--- Next Message ---\n\n"
MODEL = ModelName.GEMINI_FLASH


def load_content(content: Any) -> Any:
    if not isinstance(content, (str, Path)):
        return content
    else:
        try:
            return Path(content).read_text()
        except Exception as e:
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
    messages: Messages, system: str = "", model_name: str = MODEL, joiner: Content = MERGE_JOINER
) -> dict:
    kwargs = {"messages": deepcopy(messages)}
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
    system: Content
    messages: Messages = Field(default_factory=list)
    model_name: str = MODEL


@action.pydantic(reads=["system", "messages", "model_name"], writes=["messages"])
def add_assistant_message(state: Dialog, creator: Content) -> Dialog:
    try:
        res = (
            creator(
                **messages_to_kwargs(
                    messages=deepcopy(state.messages), system=state.system, model_name=state.model_name
                )
            )
            if callable(creator)
            else creator
        )
    except Exception as e:
        logger.exception(f"Error in respond. Last message: {state.messages[-1]}")
        res = f"Error: {e}"
    state.messages.append(assistant_message(content=res))
    return state


@action.pydantic(reads=[], writes=["messages"])
def add_user_message(state: Dialog, content: Content, template: Content = "") -> Dialog:
    state.messages.append(user_message(content=content, template=template))
    return state


@action(reads=[], writes=[])
def say_hello(state: Dialog) -> Dialog:
    logger.success("Hello!")
    return state


creator = partial(
    instructor.from_gemini(client=GenerativeModel(model_name=MODEL), mode=instructor.Mode.GEMINI_JSON).create,
    response_model=str,
)

tracker = LocalTrackingClient(project="stats")
app = (
    ApplicationBuilder()
    .with_actions(add_user_message.bind(template=""), add_assistant_message.bind(creator=creator))  # type:ignore
    .with_typing(PydanticTypingSystem(Dialog))
    # .with_state(Dialog(system="../../stats_system.txt"))
    # .with_entrypoint("add_user_message")
    .with_transitions(("add_user_message", "add_assistant_message"), ("add_assistant_message", "add_user_message"))
    .with_tracker(tracker=tracker)  # type:ignore
    .with_identifiers(app_id="stats_3")
    .initialize_from(
        initializer=tracker,
        resume_at_next_action=True,
        default_state={
            "system": load_content("../../stats_system.txt"),
            "messages": load_messages("../../stats_start.json"),
        },
        default_entrypoint="add_user_message",
        fork_from_app_id="stats_2",
        fork_from_sequence_id=3,
    )
    .build()
)
# logger.info(app.state)
for prompt in Path("../../stats_prompts").glob("*.txt"):
    while True:
        step_result = app.step(inputs={"content": load_content(prompt)})
        if step_result is None:
            logger.error("app.step returned None")
            break
        action, result, state = step_result
        logger.info(f"Action: {action.name}")
        if action.name == "add_assistant_message":
            break
