from uuid import uuid4

from pydantic import UUID4, BaseModel, Field, field_validator

from ayy.agent import (
    MAX_MESSAGE_TOKENS,
    MERGE_JOINER,
    Agent,
    Content,
    Messages,
    MessageType,
    agent_to_kwargs,
    get_trim_index,
    summarize_messages,
    user_message,
)
from ayy.dialog import chat_message
from ayy.memory import Summary, summary_to_messages


class Task(BaseModel):
    id: UUID4 = Field(default_factory=lambda: uuid4())
    name: str = ""
    agent_id: UUID4
    task_query: str
    available_tools_message: MessageType = Field(default_factory=dict)
    recommended_tools_message: MessageType = Field(default_factory=dict)
    selected_tools_message: MessageType = Field(default_factory=dict)
    summary: Summary | None = None
    summarized_task_tools: list[int] = Field(default_factory=list)

    @field_validator("summarized_task_tools")
    @classmethod
    def validate_summarized_task_tools(cls, v: list[int]) -> list[int]:
        return sorted(set(v))


class Tool(BaseModel):
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


class TaskTool(BaseModel):
    id: int
    task_id: UUID4
    position: int
    tool: Tool
    tool_args_messages: Messages = Field(default_factory=list)
    tool_result_messages: Messages = Field(default_factory=list)
    used: bool = False


def task_to_messages(task: Task, task_tools: list[TaskTool] | None = None, summarized: bool = False) -> Messages:
    message_fields = [task.available_tools_message, task.recommended_tools_message, task.selected_tools_message]
    messages = [user_message(content=task.task_query)] + [msg for msg in message_fields if msg]
    if summarized and task.summary is not None:
        messages += summary_to_messages(task.summary)
    if task_tools:
        messages += [
            message
            for task_tool in task_tools
            for message in task_tool_to_messages(task_tool)
            if task_tool.id not in (task.summarized_task_tools if summarized else [])
        ]
    return [{**msg, "task_id": task.id} for msg in messages]


def task_tool_to_messages(task_tool: TaskTool) -> Messages:
    if task_tool.tool.name == "get_selected_tools":
        return []
    messages = []
    if task_tool.tool.prompt:
        role = "assistant" if task_tool.tool.name == "ask_user" else "user"
        messages.append(chat_message(role=role, content=task_tool.tool.prompt, task_tool_id=task_tool.id))
    if task_tool.tool_args_messages:
        messages += [{**msg, "task_tool_id": task_tool.id} for msg in task_tool.tool_args_messages]
    if task_tool.tool_result_messages:
        messages += [{**msg, "task_tool_id": task_tool.id} for msg in task_tool.tool_result_messages]
    return messages


def task_to_kwargs(
    task: Task,
    task_tools: list[TaskTool],
    agent: Agent,
    messages: Messages | None = None,
    max_message_tokens: int | None = None,
    joiner: Content = MERGE_JOINER,
    summarizer_agent: Agent | None = None,
) -> dict:
    messages = messages or []
    messages = [msg for msg in messages if msg["role"] != "system"]
    if summarizer_agent is None:
        return agent_to_kwargs(
            agent=agent,
            messages=task_to_messages(task=task, task_tools=task_tools, summarized=True) + messages,
            joiner=joiner,
        )
    task_tools_messages = [
        message
        for task_tool in task_tools
        for message in task_tool_to_messages(task_tool)
        if task_tool.id not in task.summarized_task_tools
    ]
    trim_idx = get_trim_index(
        messages=task_tools_messages,
        max_message_tokens=max_message_tokens or agent.max_message_tokens or MAX_MESSAGE_TOKENS,
    )
    if trim_idx:
        summary_message = (
            [
                user_message(
                    content=f"<summary_of_our_previous_conversation(s)>\n{task.summary.summary_str()}\n</summary_of_our_previous_conversation(s)>"
                )
            ]
            if task.summary
            else []
        )
        summary = summarize_messages(
            messages=summary_message + task_tools_messages[:trim_idx],
            summarizer_agent=summarizer_agent,
            joiner=joiner,
        )
        if summary:
            task.summary = summary
            task.summarized_task_tools = sorted(
                set(
                    task.summarized_task_tools
                    + [
                        message["task_tool_id"]
                        for message in task_tools_messages[:trim_idx]
                        if message.get("task_tool_id", None) is not None
                    ]
                )
            )
    return agent_to_kwargs(
        agent=agent,
        messages=task_to_messages(task=task, task_tools=task_tools, summarized=True) + messages,
        joiner=joiner,
    )
