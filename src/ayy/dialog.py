import json
from copy import deepcopy
from enum import StrEnum
from functools import partial
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Annotated, Any, Self, cast
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

MAX_MESSAGE_TOKENS = 100_000
MERGE_JOINER = "\n\n--- Next Message ---\n\n"
TEMPERATURE = 0.1
MAX_TOKENS = 3000
DEFAULT_TAG = "RECALL"


class ModelName(StrEnum):
    GPT = "gpt-4o-2024-11-20"
    GPT_MINI = "gpt-4o-mini"
    HAIKU = "claude-3-haiku-latest"
    SONNET = "claude-3-5-sonnet-latest"
    OPUS = "claude-3-opus-latest"
    GEMINI_PRO = "gemini-1.5-pro-001"
    GEMINI_FLASH = "gemini-1.5-flash-002"
    GEMINI_FLASH_EXP = "gemini-2.0-flash-exp"


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


MODEL_NAME = ModelName.GEMINI_FLASH
Content = Annotated[Any, AfterValidator(load_content)]
MessageType = dict[str, Content]


class MessagePurpose(StrEnum):
    SUMMARY = "summary"
    CONVO = "convo"
    TOOL = "tool"
    AVAILABLE_TOOLS = "available_tools"
    RECOMMENDED_TOOLS = "recommended_tools"
    SELECTED_TOOLS = "selected_tools"
    ERROR = "error"


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
    role: str,
    content: Content,
    template: Content = "",
    purpose: MessagePurpose = DEFAULT_MESSAGE_PURPOSE,
    **kwargs,
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
    return {"role": role, "content": message_content, "purpose": purpose, **kwargs}


def system_message(
    content: Content, template: Content = "", purpose: MessagePurpose = DEFAULT_MESSAGE_PURPOSE, **kwargs
) -> MessageType:
    return chat_message(role="system", content=content, template=template, purpose=purpose, **kwargs)


def user_message(
    content: Content, template: Content = "", purpose: MessagePurpose = DEFAULT_MESSAGE_PURPOSE, **kwargs
) -> MessageType:
    return chat_message(role="user", content=content, template=template, purpose=purpose, **kwargs)


def assistant_message(
    content: Content, template: Content = "", purpose: MessagePurpose = DEFAULT_MESSAGE_PURPOSE, **kwargs
) -> MessageType:
    return chat_message(role="assistant", content=content, template=template, purpose=purpose, **kwargs)


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


class DialogToolSignature(BaseModel):
    name: str = ""
    signature: str = ""
    docstring: str = ""
    system: str = ""


class Dialog(BaseModel):
    id: UUID4 = Field(default_factory=lambda: uuid4())
    name: str = ""
    system: Content = ""
    summary: Summary | None = None
    messages: Messages = Field(default_factory=list)
    model_name: ModelName = MODEL_NAME
    creation_config: dict = dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    max_message_tokens: int | None = None
    dialog_tool_signature: dict = Field(default_factory=dict)
    available_tools: list[str] = Field(default_factory=list)
    include_tool_guidelines: bool = True

    @model_validator(mode="after")
    def validate_signature(self) -> Self:
        self.name = self.dialog_tool_signature.get("name", self.name).strip()
        self.system = self.dialog_tool_signature.get("system", self.system).strip()
        if self.include_tool_guidelines:
            self.system += f"\n\n<tool_selection_guidelines>\n{SELECT_TOOLS}\n</tool_selection_guidelines>"
        self.system = self.system.strip()
        if "system" in self.dialog_tool_signature:
            del self.dialog_tool_signature["system"]
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
            self.max_message_tokens = 100_000
        elif "claude" in self.model_name.lower():
            self.max_message_tokens = 175_000
        elif "gemini" in self.model_name.lower():
            self.max_message_tokens = 800_000
        return self


def dialog_to_messages(dialog: Dialog) -> Messages:
    return [system_message(content=dialog.system)] + dialog.messages


class Task(BaseModel):
    id: UUID4 = Field(default_factory=lambda: uuid4())
    name: str = ""
    dialog_id: UUID4
    available_tools_message: MessageType = Field(default_factory=dict)
    recommended_tools_message: MessageType = Field(default_factory=dict)
    selected_tools_message: MessageType = Field(default_factory=dict)
    messages: Messages = Field(default_factory=list)
    trim_idx: int = 0


class TaskTool(BaseModel):
    id: int
    position: int
    task_id: UUID4
    tool: Tool
    args_message: MessageType = Field(default_factory=dict)
    result_message: MessageType = Field(default_factory=dict)
    used: bool = False


def task_to_messages(task: Task) -> Messages:
    message_fields = [task.available_tools_message, task.recommended_tools_message, task.selected_tools_message]
    return [msg for msg in message_fields if msg]


def task_tool_to_messages(task_tool: TaskTool) -> Messages:
    messages = []
    if task_tool.tool.prompt:
        messages.append(
            user_message(
                content=task_tool.tool.prompt,
                purpose=MessagePurpose.CONVO
                if task_tool.tool.name == "get_selected_tools"
                else MessagePurpose.TOOL,
                task_tool_id=task_tool.id,
            )
        )
    if task_tool.args_message:
        task_tool.args_message["task_tool_id"] = task_tool.id
        messages.append(task_tool.args_message)
    if task_tool.result_message:
        task_tool.result_message["task_tool_id"] = task_tool.id
        messages.append(task_tool.result_message)
    return messages


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


def trim_messages(
    messages: Messages, max_message_tokens: int | None = MAX_MESSAGE_TOKENS
) -> tuple[list[MessageType], list[MessageType]]:
    if not messages or max_message_tokens is None:
        return messages, []
    system_msg = [messages[0]] if messages and messages[0]["role"] == "system" else []
    working_messages = messages[1:] if system_msg else messages[:]
    if working_messages and working_messages[0]["role"] == "assistant":
        working_messages[0] = {
            **working_messages[0],
            "role": "user",
            "content": f"<previous_assistant_message>\n{working_messages[0]['content']}\n</previous_assistant_message>",
        }
    total_tokens = 0
    trim_idx = len(working_messages) - 1
    while trim_idx >= 0:
        msg_tokens = count_tokens([working_messages[trim_idx]])
        if total_tokens + msg_tokens > max_message_tokens:
            break
        total_tokens += msg_tokens
        trim_idx -= 1
    trimmed_messages = working_messages[: trim_idx + 1]
    if (
        trimmed_messages
        and trimmed_messages[-1].get("purpose", MessagePurpose.CONVO) == MessagePurpose.AVAILABLE_TOOLS
    ):
        trim_idx = max(-1, trim_idx - 1)
    trimmed_messages = working_messages[: trim_idx + 1]
    if (
        len(trimmed_messages) == 1
        and trimmed_messages[0].get("purpose", MessagePurpose.CONVO) == MessagePurpose.SUMMARY
    ):
        trim_idx = max(-1, trim_idx - 1)
    return system_msg + working_messages[trim_idx + 1 :], [
        message
        for message in working_messages[: trim_idx + 1]
        if message.get("purpose", MessagePurpose.CONVO) == MessagePurpose.CONVO
    ]


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


def task_to_kwargs(
    task: Task,
    task_tools: list[TaskTool],
    dialog: Dialog,
    max_message_tokens: int | None = None,
    joiner: Content = MERGE_JOINER,
) -> dict:
    task_tools_messages = flatten_list([task_tool_to_messages(task_tool) for task_tool in task_tools])[
        task.trim_idx :
    ]
    trim_idx = get_trim_index(
        messages=task_tools_messages,
        max_message_tokens=dialog.max_message_tokens or max_message_tokens or MAX_MESSAGE_TOKENS,
    )
    task.trim_idx = trim_idx
    task_tools_messages = task_tools_messages[trim_idx:]
    return dialog_to_kwargs(dialog=dialog, messages=task_to_messages(task) + task_tools_messages, joiner=joiner)


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


def dialog_to_kwargs(dialog: Dialog, messages: Messages | None = None, joiner: Content = MERGE_JOINER) -> dict:
    messages = messages or []
    kwargs = messages_to_kwargs(
        messages=dialog.messages + [msg for msg in messages if msg["role"] != "system"],
        system=dialog.system,
        model_name=dialog.model_name,
        joiner=joiner,
    )
    if "gemini" in dialog.model_name.lower():
        kwargs["generation_config"] = dialog.creation_config
    else:
        kwargs.update(dialog.creation_config)
    return kwargs


def summarize_messages(
    messages: Messages, summarizer_dialog: Dialog | None = None, max_message_tokens: int = MAX_MESSAGE_TOKENS
) -> tuple[Messages, Summary | None]:
    # return messages, None
    if summarizer_dialog is None:
        return messages, None
    messages = deepcopy(messages)
    messages, trimmed_messages = trim_messages(messages=messages, max_message_tokens=max_message_tokens)
    logger.info(
        f"Called trim_messages with max_message_tokens={max_message_tokens}. Got messages: {messages[-2:]}, trimmed_messages: {trimmed_messages[-2:]}"
    )
    if not trimmed_messages:
        return messages, None
    logger.info(f"Summarizing messages: {trimmed_messages[-2:]}")
    try:
        creator = create_creator(model_name=summarizer_dialog.model_name)
        summary: Summary = creator.create(
            **dialog_to_kwargs(
                dialog=summarizer_dialog,
                messages=trimmed_messages + [user_message(content="Summarize the conversation so far.")],
            ),
            response_model=Summary,  # type: ignore
        )
        episodic_explanation = """
Episodic memories are relevant to ongoing tasks. They are memories of the current context and parameters, intermediate steps or results, short-term preferences or needs, time-sensitive details, and recent interactions or decisions.
"""
        summary_message = user_message(
            content=f"<summary_of_our_previous_conversation(s)>\n{summary.summary_str(semantic=False, episodic=True)}\n</summary_of_our_previous_conversation(s)>\n\n{episodic_explanation}",
            purpose=MessagePurpose.SUMMARY,
        )
        if messages and messages[0]["role"] == "system":
            messages = [messages[0], summary_message] + messages[1:]
        else:
            messages = [summary_message] + messages
        logger.success("Summarized messages")
        return messages, summary
    except Exception:
        logger.exception(f"Could not summarize messages: {trimmed_messages[-2:]}")
        return messages, None


def add_message(
    task_or_dialog: Task | Dialog,
    role: str = "user",
    content: Content = "",
    template: Content = "",
    purpose: MessagePurpose = DEFAULT_MESSAGE_PURPOSE,
    message: MessageType | None = None,
    summarizer_dialog: Dialog | None = None,
) -> Task | Dialog:
    if content is None:
        return task_or_dialog
    max_message_tokens = (
        task_or_dialog.max_message_tokens if isinstance(task_or_dialog, Dialog) else MAX_MESSAGE_TOKENS
    )
    task_or_dialog.messages = summarize_messages(
        messages=task_or_dialog.messages
        + [message or chat_message(role=role, content=content, template=template, purpose=purpose)],
        summarizer_dialog=summarizer_dialog,
        max_message_tokens=max_message_tokens or MAX_MESSAGE_TOKENS,
    )[0]
    return task_or_dialog


def add_dialog_message(
    dialog: Dialog,
    role: str = "user",
    content: Content = "",
    template: Content = "",
    purpose: MessagePurpose = DEFAULT_MESSAGE_PURPOSE,
    message: MessageType | None = None,
    summarizer_dialog: Dialog | None = None,
) -> Dialog:
    return cast(
        Dialog,
        add_message(
            task_or_dialog=dialog,
            role=role,
            content=content,
            template=template,
            purpose=purpose,
            message=message,
            summarizer_dialog=summarizer_dialog,
        ),
    )


def add_task_message(
    task: Task,
    role: str = "user",
    content: Content = "",
    template: Content = "",
    purpose: MessagePurpose = DEFAULT_MESSAGE_PURPOSE,
    message: MessageType | None = None,
) -> Task:
    return cast(
        Task,
        add_message(
            task_or_dialog=task, role=role, content=content, template=template, purpose=purpose, message=message
        ),
    )
