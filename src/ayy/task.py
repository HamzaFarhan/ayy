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


class Task(BaseModel):
    id: UUID4 = Field(default_factory=lambda: uuid4())
    name: str = ""
    agent_id: UUID4
    available_tools_message: MessageType = Field(default_factory=dict)
    recommended_tools_message: MessageType = Field(default_factory=dict)
    selected_tools_message: MessageType = Field(default_factory=dict)
    summarized_task_tools: list[int] = Field(default_factory=list)

    @field_validator("summarized_task_tools")
    @classmethod
    def validate_summarized_task_tools(cls, v: list[int]) -> list[int]:
        return sorted(set(v))


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


class TaskTool(BaseModel):
    id: int
    task_id: UUID4
    position: int
    tool: Tool
    tool_args_message: MessageType = Field(default_factory=dict)
    tool_result_message: MessageType = Field(default_factory=dict)
    used: bool = False


def task_to_messages(task: Task) -> Messages:
    message_fields = [task.available_tools_message, task.recommended_tools_message, task.selected_tools_message]
    return [msg for msg in message_fields if msg]


def task_tool_to_messages(task_tool: TaskTool) -> Messages:
    messages = []
    if task_tool.tool.prompt:
        messages.append(user_message(content=task_tool.tool.prompt, task_tool_id=task_tool.id))
    if task_tool.tool_args_message:
        task_tool.tool_args_message["task_tool_id"] = task_tool.id
        messages.append(task_tool.tool_args_message)
    if task_tool.tool_result_message:
        task_tool.tool_result_message["task_tool_id"] = task_tool.id
        messages.append(task_tool.tool_result_message)
    return messages


def task_to_kwargs(
    task: Task,
    task_tools: list[TaskTool],
    agent: Agent,
    max_message_tokens: int | None = None,
    joiner: Content = MERGE_JOINER,
    summarizer_agent: Agent | None = None,
) -> dict:
    task_tools_messages = [
        message
        for task_tool in task_tools
        for message in task_tool_to_messages(task_tool)
        if task_tool.id not in task.summarized_task_tools
    ]
    messages = (agent.summary.model_dump().get("messages", []) if agent.summary else []) + task_tools_messages
    trim_idx = get_trim_index(
        messages=messages,
        max_message_tokens=max_message_tokens or agent.max_message_tokens or MAX_MESSAGE_TOKENS,
    )
    if trim_idx:
        messages_to_summarize = task_to_messages(task) + messages[:trim_idx]
        summary = summarize_messages(messages=messages_to_summarize, summarizer_agent=summarizer_agent)
        if summary:
            if not agent.summary:
                agent.summary = summary
            else:
                agent.summary.messages.extend(summary.messages)
                agent.summary.semantic_memories.extend(summary.semantic_memories)
            task.summarized_task_tools = sorted(
                set(
                    task.summarized_task_tools
                    + [
                        message["task_tool_id"]
                        for message in messages[:trim_idx]
                        if message.get("task_tool_id", None) is not None
                    ]
                )
            )
    return agent_to_kwargs(agent=agent, messages=task_to_messages(task) + messages[trim_idx:], joiner=joiner)
