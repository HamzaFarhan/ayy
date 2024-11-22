import inspect
from collections import deque
from functools import partial
from typing import Deque, Literal

import dill
from instructor import AsyncInstructor, Instructor
from loguru import logger
from pydantic import BaseModel, Field, create_model
from valkey import Valkey

from ayy import tools
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


def get_tool_queue(valkey_client: Valkey) -> Deque:
    queue = valkey_client.get("tool_queue")
    return deque(dill.loads(queue)) if queue else deque()  # type: ignore


def get_current_tool(valkey_client: Valkey) -> str:
    return valkey_client.get("current_tool_name") or DEFAULT_TOOL.name  # type: ignore


def update_tool_queue(valkey_client: Valkey, tool_queue: Deque):
    tools = []
    for tool in tool_queue:
        if isinstance(tool, Tool):
            tools.append(Tool(**tool.model_dump()))
        else:
            tools.append(tool)
    valkey_client.set("tool_queue", dill.dumps(tools))


def pop_next_tool(valkey_client: Valkey) -> Tool:
    tool_queue = get_tool_queue(valkey_client)
    tool = tool_queue.popleft()
    update_tool_queue(valkey_client=valkey_client, tool_queue=tool_queue)
    return tool


def update_current_tool(valkey_client: Valkey, tool_name: str):
    valkey_client.set("current_tool_name", tool_name)


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
    try:
        tool_attr = getattr(tools, tool.name, globals().get(tool.name, None))
        if tool_attr is None:
            raise ValueError(f"Tool '{tool.name}' not found in tools module")
        if not inspect.isfunction(tool_attr):
            raise ValueError(f"Tool '{tool.name}' is not a function.\nGot {type(tool_attr).__name__} instead")
        selected_tool = tool_attr
    except AttributeError:
        raise ValueError(f"Tool '{tool.name}' not found in tools module")
    if tool.prompt:
        dialog.messages.append(user_message(content=tool.prompt))

    tool_type = getattr(tools, "tool_types", globals().get("tool_types", {})).get(tool.name, None)
    all_tools = inspect.getmembers(tools, inspect.isfunction)
    if tool.name == "get_selected_tools" and all_tools:
        selected_tool = partial(get_selected_tools, valkey_client)
        tool_type = list[
            create_model(
                "SelectedTool",
                name=(Literal[*[tool_member[0] for tool_member in all_tools]], ...),  # type: ignore
                __base__=Tool,
            )
        ]
    if tool_type is None:
        tool_type = function_to_type(
            func=selected_tool,
            ignore_default_values=ignore_default_values,
            skip_default_params=skip_default_params,
        )
    logger.info(f"\n\nCalling {tool.name} with messages: {dialog.messages}\n\n")
    creator_res = creator.create(
        **dialog_to_kwargs(dialog=dialog),
        response_model=tool_type,  # type: ignore
    )
    logger.info(f"{tool.name} creator_res: {creator_res}")
    logger.info(f"{tool.name} selected_tool: {selected_tool}")
    if isinstance(creator_res, BaseModel):
        res = selected_tool(**creator_res.model_dump())
    else:
        res = selected_tool(creator_res)  # type: ignore
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
        dialog = run_tool(valkey_client=valkey_client, creator=creator, dialog=dialog, tool=tool)
    return dialog


def run_next_tool(valkey_client: Valkey, creator: Instructor | AsyncInstructor, dialog: Dialog) -> Dialog:
    tool_queue = get_tool_queue(valkey_client)
    if tool_queue:
        tool = pop_next_tool(valkey_client=valkey_client)
        dialog = run_selected_tool(valkey_client=valkey_client, creator=creator, dialog=dialog, tool=tool)
    return dialog


def run_tools(
    valkey_client: Valkey, creator: Instructor | AsyncInstructor, dialog: Dialog, continue_dialog: bool = True
) -> Dialog:
    tool_queue = get_tool_queue(valkey_client)
    current_tool_name = get_current_tool(valkey_client)
    if not tool_queue:
        tool_queue = deque([DEFAULT_TOOL])
        update_tool_queue(valkey_client=valkey_client, tool_queue=tool_queue)

    while tool_queue:
        print(f"\nTOOL QUEUE: {tool_queue}\n")
        current_tool = pop_next_tool(valkey_client=valkey_client)

        if not isinstance(current_tool, Tool) and callable(current_tool):
            current_tool_name = (
                current_tool.__name__ if not isinstance(current_tool, partial) else current_tool.func.__name__
            )
            update_current_tool(valkey_client=valkey_client, tool_name=current_tool_name)
            res = current_tool()
            if res:
                if isinstance(res, Dialog):
                    dialog = res
                else:
                    dialog.messages.append(assistant_message(content=str(res)))
            continue

        current_tool_name = current_tool.name
        update_current_tool(valkey_client=valkey_client, tool_name=current_tool_name)
        dialog = run_selected_tool(valkey_client=valkey_client, creator=creator, dialog=dialog, tool=current_tool)
        tool_queue = get_tool_queue(valkey_client)
        update_tool_queue(valkey_client=valkey_client, tool_queue=tool_queue)

    if continue_dialog:
        current_tool_name = get_current_tool(valkey_client=valkey_client)
        seq = int(current_tool_name == "ask_user")
        while True:
            if seq % 2 == 0 or current_tool_name == "call_ai":
                user_input = input("('q' or 'exit' or 'quit' to quit) > ")
                if user_input.lower() in ["q", "exit", "quit"]:
                    break
                dialog = run_tools(
                    valkey_client=valkey_client,
                    creator=creator,
                    dialog=new_task(valkey_client=valkey_client, dialog=dialog, task=user_input),
                    continue_dialog=False,
                )
            else:
                current_tool_name = "call_ai"
                update_current_tool(valkey_client=valkey_client, tool_name=current_tool_name)
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
    update_tool_queue(valkey_client=valkey_client, tool_queue=tool_queue)


def new_task(valkey_client: Valkey, dialog: Dialog, task: str) -> Dialog:
    tool_queue = get_tool_queue(valkey_client)

    tools_info = "\n\n".join(
        [
            f"Tool {i}:\n{get_function_info(func)}"
            for i, (_, func) in enumerate(inspect.getmembers(tools, inspect.isfunction), start=1)
        ]
    )
    dialog.messages += [user_message(content=f"Available tools for this task:\n{tools_info}")]
    tool_queue.appendleft(Tool(chain_of_thought="", name="get_selected_tools", prompt=task))
    update_tool_queue(valkey_client=valkey_client, tool_queue=tool_queue)
    return dialog
