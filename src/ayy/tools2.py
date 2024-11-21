import json
from collections import deque
from functools import partial
from typing import Any, Callable, Literal, Deque

from instructor import AsyncInstructor, Instructor
from loguru import logger
from pydantic import BaseModel, Field, create_model
from valkey import Valkey

from ayy.dialog import Dialog, ModelName, assistant_message, dialog_to_kwargs, user_message
from ayy.func_utils import function_to_type, get_function_info

MODEL_NAME = ModelName.GEMINI_FLASH
DEFAULT_PROMPT = "Generate a response if you've been asked. Otherwise, ask the user how they are doing."


class Tool(BaseModel):
    chain_of_thought: str
    name: str
    prompt: str = Field(
        ...,
        description="An LLM will receive the messages so far and the tools calls and results up until now. This prompt will then be used to ask the LLM to generate arguments for the selected tool based on the tool's signature. If the tool doesn't have any parameters, then it doesn't need a prompt.",
    )


DEFAULT_TOOL = Tool(chain_of_thought="", name="call_ai", prompt=DEFAULT_PROMPT)


def get_tool_dict(valkey_client: Valkey) -> dict:
    """Get tool dictionary from Valkey store"""
    tool_dict = valkey_client.get("tool_dict")
    return json.loads(tool_dict) if tool_dict else {}  # type: ignore


def get_tool_queue(valkey_client: Valkey) -> Deque:
    """Get tool queue from Valkey store"""
    queue = valkey_client.get("tool_queue")
    return deque(queue) if queue else deque()  # type: ignore


def get_current_tool(valkey_client: Valkey) -> str:
    """Get current tool name from Valkey store"""
    return valkey_client.get("current_tool_name") or DEFAULT_TOOL.name  # type: ignore


def update_tool_dict(valkey_client: Valkey, tool_dict: dict):
    """Update tool dictionary in Valkey store"""
    valkey_client.set("tool_dict", json.dumps(tool_dict))


def update_tool_queue(valkey_client: Valkey, queue: Deque):
    """Update tool queue in Valkey store"""
    valkey_client.set("tool_queue", list(queue))  # type: ignore


def update_current_tool(valkey_client: Valkey, tool_name: str):
    """Update current tool name in Valkey store"""
    valkey_client.set("current_tool_name", tool_name)


def call_ai(inputs: Any) -> Any:
    "Not a pre-defined tool."
    return inputs


def ask_user(prompt: str) -> str:
    "Prompt the user for a response"
    return prompt


def run_call_ai(creator: Instructor | AsyncInstructor, dialog: Dialog, tool: Tool) -> Dialog:
    tool.prompt = tool.prompt or DEFAULT_PROMPT
    dialog.messages.append(user_message(content=tool.prompt))
    logger.info(f"\n\nCalling AI with messages: {dialog.messages}\n\n")
    res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
    logger.success(f"call_ai result: {res}")
    dialog.messages.append(assistant_message(content=res))
    return dialog


def call_ask_user(dialog: Dialog, tool: Tool) -> Dialog:
    tool.prompt = tool.prompt or DEFAULT_PROMPT
    res = input(f"{tool.prompt}\n> ")
    dialog.messages += [assistant_message(content=tool.prompt), user_message(content=res)]
    return dialog


def run_tool(
    valkey_client: Valkey,
    creator: Instructor | AsyncInstructor,
    dialog: Dialog,
    tool: Tool,
    ignore_default_values: bool = False,
    skip_default_params: bool = False,
) -> Dialog:
    tool_dict = get_tool_dict(valkey_client)

    if tool.prompt:
        dialog.messages.append(user_message(content=tool.prompt))
    logger.info(f"\n\nCalling for {tool.name} with messages: {dialog.messages}\n\n")
    creator_res = creator.create(
        **dialog_to_kwargs(dialog=dialog),
        response_model=tool_dict[tool.name].get(
            "type",
            function_to_type(
                func=tool_dict[tool.name]["func"],
                ignore_default_values=ignore_default_values,
                skip_default_params=skip_default_params,
            ),
        ),
    )
    logger.info(f"{tool.name} creator_res: {creator_res}")
    selected_tool = tool_dict[tool.name]["func"]
    if isinstance(creator_res, BaseModel):
        res = selected_tool(**creator_res.model_dump())
    else:
        res = selected_tool(creator_res)
    logger.success(f"{tool.name} result: {res}")
    if isinstance(res, Dialog):
        return res
    dialog.messages.append(assistant_message(content=str(res)))
    return dialog


def run_selected_tool(
    valkey_client: Valkey, creator: Instructor | AsyncInstructor, dialog: Dialog, tool: Tool
) -> Dialog:
    if tool.name.lower() == "ask_user":
        dialog = call_ask_user(dialog=dialog, tool=tool)
    elif tool.name.lower() == "call_ai":
        dialog = run_call_ai(creator=creator, dialog=dialog, tool=tool)
    else:
        dialog = run_tool(valkey_client, creator=creator, dialog=dialog, tool=tool)
    return dialog


def run_next_tool(valkey_client: Valkey, creator: Instructor | AsyncInstructor, dialog: Dialog) -> Dialog:
    tool_queue = get_tool_queue(valkey_client)
    if tool_queue:
        tool = tool_queue.popleft()
        update_tool_queue(valkey_client, tool_queue)
        dialog = run_selected_tool(valkey_client, creator=creator, dialog=dialog, tool=tool)
    return dialog


def run_tools(
    valkey_client: Valkey, creator: Instructor | AsyncInstructor, dialog: Dialog, continue_dialog: bool = True
) -> Dialog:
    tool_queue = get_tool_queue(valkey_client)
    current_tool_name = get_current_tool(valkey_client)
    if not tool_queue:
        tool_queue = deque([DEFAULT_TOOL])
        update_tool_queue(valkey_client, tool_queue)

    while tool_queue:
        print(f"\nTOOL QUEUE: {tool_queue}\n")
        print(f"\nTOOL DICT: {get_tool_dict(valkey_client)}\n")
        current_tool = tool_queue.popleft()
        update_tool_queue(valkey_client, tool_queue)

        if not isinstance(current_tool, Tool) and callable(current_tool):
            current_tool_name = (
                current_tool.__name__ if not isinstance(current_tool, partial) else current_tool.func.__name__
            )
            update_current_tool(valkey_client, current_tool_name)
            res = current_tool()
            if res:
                if isinstance(res, Dialog):
                    dialog = res
                else:
                    dialog.messages.append(assistant_message(content=str(res)))
            continue

        current_tool_name = current_tool.name
        update_current_tool(valkey_client, current_tool_name)
        dialog = run_selected_tool(valkey_client, creator=creator, dialog=dialog, tool=current_tool)
        tool_queue = get_tool_queue(valkey_client)

    if continue_dialog:
        current_tool_name = get_current_tool(valkey_client)
        seq = int(current_tool_name == "ask_user")
        while True:
            if seq % 2 == 0 or current_tool_name == "call_ai":
                user_input = input("('q' or 'exit' or 'quit' to quit) > ")
                if user_input.lower() in ["q", "exit", "quit"]:
                    break
                tool_queue.appendleft(
                    partial(new_task, valkey_client=valkey_client, dialog=dialog, task=user_input)  # type: ignore
                )
                update_tool_queue(valkey_client, tool_queue)
                dialog = run_tools(valkey_client, creator=creator, dialog=dialog, continue_dialog=False)
            else:
                current_tool_name = "call_ai"
                update_current_tool(valkey_client, current_tool_name)
                res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
                logger.success(f"ai response: {res}")
                dialog.messages.append(assistant_message(content=res))
            seq += 1

    logger.success(f"Messages: {dialog.messages[-2:]}")
    return dialog


def get_selected_tools(valkey_client: Valkey, selected_tools: list[Tool]):
    """Get a list of selected tools for the task"""
    tool_queue = get_tool_queue(valkey_client)
    tool_queue.extendleft(selected_tools[::-1])
    update_tool_queue(valkey_client, tool_queue)


def add_new_tools(valkey_client: Valkey, new_tools: set[Callable] | list[Callable]):
    tool_dict = get_tool_dict(valkey_client)

    for func in new_tools:
        tool_dict[func.__name__] = {"info": get_function_info(func), "func": func}

    tool_dict["get_selected_tools"] = {
        "info": get_function_info(get_selected_tools),
        "func": get_selected_tools,
        "type": list[create_model("SelectedTool", name=(Literal[*tool_dict.keys()], ...), __base__=Tool)],  # type: ignore
    }

    update_tool_dict(valkey_client, tool_dict)


def new_task(valkey_client: Valkey, dialog: Dialog, task: str, available_tools: list[str] | None = None) -> Dialog:
    tool_dict = get_tool_dict(valkey_client)
    tool_queue = get_tool_queue(valkey_client)

    tools_info = "\n\n".join(
        [
            f"Tool {i}:\n{tool_dict[tool]['info']}"
            for i, tool in enumerate(available_tools or tool_dict.keys(), start=1)
        ]
    )
    dialog.messages += [user_message(content=f"Available tools for this task:\n{tools_info}")]
    tool_queue.appendleft(Tool(chain_of_thought="", name="get_selected_tools", prompt=task))
    update_tool_queue(valkey_client, tool_queue)
    return dialog


# Initialize default tools
DEFAULT_TOOLS = {call_ai, ask_user}
