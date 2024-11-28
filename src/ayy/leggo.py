import inspect
from collections import deque
from functools import partial
from typing import Literal

from instructor import AsyncInstructor, Instructor
from loguru import logger
from pydantic import BaseModel, create_model
from valkey import Valkey

from ayy import tools
from ayy.dialog import (
    DEFAULT_PROMPT,
    Dialog,
    ModelName,
    add_message,
    assistant_message,
    dialog_to_kwargs,
    user_message,
)
from ayy.func_utils import function_to_type, get_function_info, get_functions_from_module
from ayy.tools import DEFAULT_TOOL, Tool
from ayy.vk import get_current_tool, get_tool_queue, pop_next_tool, update_current_tool, update_tool_queue

MODEL_NAME = ModelName.GEMINI_FLASH
PINNED_TOOLS = set(["ask_user", "call_ai"])
TAG_MESSAGES = True


def run_ask_user(dialog: Dialog, tool: Tool) -> Dialog:
    tool.prompt = tool.prompt or DEFAULT_PROMPT
    res = input(f"{tool.prompt}\n> ")
    dialog.messages += [assistant_message(content=tool.prompt), user_message(content=res)]
    return dialog


def run_call_ai(
    creator: Instructor | AsyncInstructor, dialog: Dialog, tool: Tool, tag_messages: bool = TAG_MESSAGES
) -> Dialog:
    tool.prompt = tool.prompt or DEFAULT_PROMPT
    dialog = add_message(dialog=dialog, message=user_message(content=tool.prompt))
    logger.info(f"\n\nCalling AI with messages: {dialog.messages}\n\n")
    res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
    logger.success(f"call_ai result: {res}")
    dialog = add_message(dialog=dialog, message=assistant_message(content=res), tag_messages=tag_messages)
    return dialog


def get_selected_tools(valkey_client: Valkey, selected_tools: list[Tool], get_approval: bool = True):
    """
    Get and push a list of selected tools for the task
    It will also add an ask_user at the start to approve the tools. You don't need to add it yourself.
    """
    tool_queue = get_tool_queue(valkey_client)
    tool_queue.extendleft(selected_tools[::-1])
    if get_approval:
        tool_queue_str = "\n\n".join([f"Tool {i}:\n{tool}" for i, tool in enumerate(tool_queue, start=1)])
        tool_queue.appendleft(
            Tool(
                chain_of_thought="",
                name="ask_user",
                prompt=f"I will run these tools in sequence:\n\n{tool_queue_str}\n\nDo you approve?",
            )
        )
    update_tool_queue(valkey_client=valkey_client, tool_queue=tool_queue)


def run_tool(
    valkey_client: Valkey,
    creator: Instructor | AsyncInstructor,
    dialog: Dialog,
    tool: Tool,
    ignore_default_values: bool = False,
    skip_default_params: bool = False,
    tag_messages: bool = TAG_MESSAGES,
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
        dialog = add_message(dialog=dialog, message=user_message(content=tool.prompt))

    tool_type = getattr(tools, "tool_types", globals().get("tool_types", {})).get(tool.name, None)
    all_tools = get_functions_from_module(module=tools)
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
    if isinstance(creator_res, BaseModel):
        res = selected_tool(**creator_res.model_dump())
    else:
        res = selected_tool(creator_res)  # type: ignore
    logger.success(f"{tool.name} result: {res}")
    if isinstance(res, Dialog):
        return res
    dialog = add_message(dialog=dialog, message=assistant_message(content=str(res)), tag_messages=tag_messages)
    return dialog


def run_selected_tool(
    valkey_client: Valkey, creator: Instructor | AsyncInstructor, dialog: Dialog, tool: Tool
) -> Dialog:
    if tool.name.lower() == "ask_user":
        dialog = run_ask_user(dialog=dialog, tool=tool)
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


def new_task(
    valkey_client: Valkey, dialog: Dialog, task: str, available_tools: list[str] | set[str] | None = None
) -> Dialog:
    available_tools = available_tools or []
    tool_queue = get_tool_queue(valkey_client)
    tools_info = "\n\n".join(
        [
            f"Tool {i}:\n{get_function_info(func)}"
            for i, (_, func) in enumerate(get_functions_from_module(module=tools), start=1)
            if not available_tools or func.__name__ in set(available_tools) | PINNED_TOOLS
        ]
    )
    dialog.messages += [user_message(content=f"Available tools for this task:\n{tools_info}")]
    tool_queue.appendleft(Tool(chain_of_thought="", name="get_selected_tools", prompt=task))
    update_tool_queue(valkey_client=valkey_client, tool_queue=tool_queue)
    return dialog


def run_tools(
    valkey_client: Valkey,
    creator: Instructor | AsyncInstructor,
    dialog: Dialog,
    continue_dialog: bool = True,
    available_tools: list[str] | set[str] | None = None,
    tag_messages: bool = TAG_MESSAGES,
) -> Dialog:
    tool_queue = get_tool_queue(valkey_client)
    current_tool_name = get_current_tool(valkey_client)
    if not tool_queue:
        tool_queue = deque([DEFAULT_TOOL])
        update_tool_queue(valkey_client=valkey_client, tool_queue=tool_queue)

    while tool_queue:
        tools_str = "\n\n".join([str(tool) for tool in tool_queue])
        logger.info(f"\nTOOL QUEUE:\n\n{tools_str}\n")
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
                    dialog = add_message(
                        dialog=dialog, message=assistant_message(content=str(res)), tag_messages=tag_messages
                    )
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
                    dialog=new_task(
                        valkey_client=valkey_client,
                        dialog=dialog,
                        task=user_input,
                        available_tools=available_tools,
                    ),
                    continue_dialog=False,
                )
            else:
                current_tool_name = "call_ai"
                update_current_tool(valkey_client=valkey_client, tool_name=current_tool_name)
                res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
                logger.success(f"ai response: {res}")
                dialog = add_message(
                    dialog=dialog, message=assistant_message(content=res), tag_messages=tag_messages
                )
            seq += 1

    logger.success(f"Messages: {dialog.messages[-2:]}")
    return dialog
