import inspect
from collections import deque
from functools import partial
from typing import Literal

from instructor import AsyncInstructor, Instructor
from loguru import logger
from pydantic import BaseModel, create_model

from ayy import tools
from ayy.dialog import (
    DEFAULT_PROMPT,
    DEFAULT_TOOL,
    Dialog,
    ModelName,
    Tool,
    add_message,
    assistant_message,
    dialog_to_kwargs,
    user_message,
)
from ayy.func_utils import function_to_type, get_function_info, get_functions_from_module
from ayy.torm import get_current_tool, get_tool_queue, pop_next_tool, update_current_tool, update_tool_queue

MODEL_NAME = ModelName.GEMINI_FLASH
PINNED_TOOLS = set(["ask_user", "call_ai"])
TAG_MESSAGES = True


def run_ask_user(dialog: Dialog, tool: Tool) -> Dialog:
    tool.prompt = tool.prompt or DEFAULT_PROMPT
    res = input(f"{tool.prompt}\n> ")
    dialog.messages += [assistant_message(content=tool.prompt), user_message(content=res)]
    return dialog


async def run_call_ai(
    creator: Instructor | AsyncInstructor, dialog: Dialog, tool: Tool, tag_messages: bool = TAG_MESSAGES
) -> Dialog:
    tool.prompt = tool.prompt or DEFAULT_PROMPT
    dialog = add_message(dialog=dialog, message=user_message(content=tool.prompt))
    logger.info(f"\n\nCalling AI with messages: {dialog.messages}\n\n")
    res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
    logger.success(f"call_ai result: {res}")
    dialog = add_message(dialog=dialog, message=assistant_message(content=res), tag_messages=tag_messages)
    return dialog


# async def get_selected_tools(db_name: str, selected_tools: list[Tool], get_approval: bool = False):
#     """
#     Get and push a list of selected tools for the task
#     It will also add an ask_user at the start to approve the tools. You don't need to add it yourself.
#     """
#     tool_queue = await get_tool_queue(db_name=db_name)
#     tool_queue.extendleft(selected_tools[::-1])
#     if get_approval:
#         tool_queue_str = "\n\n".join([f"Tool {i}:\n{tool}" for i, tool in enumerate(tool_queue, start=1)])
#         tool_queue.appendleft(
#             Tool(
#                 reasoning="",
#                 name="ask_user",
#                 prompt=f"I will run these tools in sequence:\n\n{tool_queue_str}\n\nDo you approve?",
#             )
#         )
#     await update_tool_queue(db_name=db_name, tool_queue=tool_queue)

async def get_selected_tools(db_name: str, dialo selected_tools: list[Tool]):
    """
    Get and push a list of selected tools for the task
    It will also add an ask_user at the start to approve the tools. You don't need to add it yourself.
    """
    tool_queue = await get_tool_queue(db_name=db_name)
    tool_queue.extendleft(selected_tools[::-1])
  
    await update_tool_queue(db_name=db_name, tool_queue=tool_queue)



async def run_tool(
    db_name: str,
    creator: Instructor | AsyncInstructor,
    dialog: Dialog,
    tool: Tool,
    ignore_default_values: bool = False,
    skip_default_params: bool = False,
    tag_messages: bool = TAG_MESSAGES,
) -> Dialog:
    is_async = False
    try:
        tool_attr = getattr(tools, tool.name, globals().get(tool.name, None))
        if tool_attr is None:
            raise ValueError(f"Tool '{tool.name}' not found in tools module")
        if not inspect.isfunction(tool_attr):
            raise ValueError(f"Tool '{tool.name}' is not a function.\nGot {type(tool_attr).__name__} instead")
        selected_tool = tool_attr
        is_async = inspect.iscoroutinefunction(tool_attr)
    except AttributeError:
        raise ValueError(f"Tool '{tool.name}' not found in tools module")
    if tool.prompt:
        dialog = add_message(dialog=dialog, message=user_message(content=tool.prompt))

    tool_type = getattr(tools, "tool_types", globals().get("tool_types", {})).get(tool.name, None)
    all_tools = get_functions_from_module(module=tools)
    if tool.name == "get_selected_tools" and len(all_tools) > 0:
        selected_tool = partial(get_selected_tools, db_name)
        tool_type = list[
            create_model(
                "SelectedTool",
                name=(Literal[*[tool_member[0] for tool_member in all_tools]], ...),  # type: ignore
                __base__=Tool,
            )
        ]
        is_async = True
    if tool_type is None:
        tool_type = function_to_type(
            func=selected_tool,
            ignore_default_values=ignore_default_values,
            skip_default_params=skip_default_params,
        )
    logger.info(f"\n\nCalling {tool.name} with messages: {dialog.messages}\n\n")
    if inspect.iscoroutinefunction(creator.create):
        creator_res = await creator.create(
            **dialog_to_kwargs(dialog=dialog),
            response_model=tool_type,  # type: ignore
        )
    else:
        creator_res = creator.create(
            **dialog_to_kwargs(dialog=dialog),
            response_model=tool_type,  # type: ignore
        )
    logger.info(f"{tool.name} creator_res: {creator_res}")
    if isinstance(creator_res, BaseModel):
        if is_async:
            res = await selected_tool(**creator_res.model_dump())
        else:
            res = selected_tool(**creator_res.model_dump())
    elif is_async:
        res = await selected_tool(creator_res)  # type: ignore
    else:
        res = selected_tool(creator_res)  # type: ignore
    logger.success(f"{tool.name} result: {res}")
    if isinstance(res, Dialog):
        return res
    dialog = add_message(dialog=dialog, message=assistant_message(content=str(res)), tag_messages=tag_messages)
    return dialog


async def run_selected_tool(
    db_name: str, creator: Instructor | AsyncInstructor, dialog: Dialog, tool: Tool
) -> Dialog:
    if tool.name.lower() == "ask_user":
        dialog = run_ask_user(dialog=dialog, tool=tool)
    elif tool.name.lower() == "call_ai":
        dialog = await run_call_ai(creator=creator, dialog=dialog, tool=tool)
    else:
        dialog = await run_tool(db_name=db_name, creator=creator, dialog=dialog, tool=tool)
    return dialog


async def run_next_tool(db_name: str, creator: Instructor | AsyncInstructor, dialog: Dialog) -> Dialog:
    tool_queue = await get_tool_queue(db_name=db_name)
    if tool_queue:
        tool = await pop_next_tool(db_name=db_name)
        dialog = await run_selected_tool(db_name=db_name, creator=creator, dialog=dialog, tool=tool)
    return dialog


async def new_task(
    db_name: str, dialog: Dialog, task: str, available_tools: list[str] | set[str] | None = None
) -> Dialog:
    available_tools = available_tools or []
    tool_queue = await get_tool_queue(db_name=db_name)
    tools_info = "\n\n".join(
        [
            f"Tool {i}:\n{get_function_info(func)}"
            for i, (_, func) in enumerate(get_functions_from_module(module=tools), start=1)
            if not available_tools or func.__name__ in set(available_tools) | PINNED_TOOLS
        ]
    )
    dialog.messages += [user_message(content=f"Available tools for this task:\n{tools_info}")]
    tool_queue.appendleft(Tool(reasoning="", name="get_selected_tools", prompt=task))
    await update_tool_queue(db_name=db_name, tool_queue=tool_queue)
    return dialog


async def run_tools(
    db_name: str,
    creator: Instructor | AsyncInstructor,
    dialog: Dialog,
    continue_dialog: bool = True,
    available_tools: list[str] | set[str] | None = None,
    tag_messages: bool = TAG_MESSAGES,
) -> Dialog:
    tool_queue = await get_tool_queue(db_name=db_name)
    current_tool = await get_current_tool(db_name=db_name)
    current_tool_name = current_tool.name
    if not tool_queue:
        tool_queue = deque([DEFAULT_TOOL])
        await update_tool_queue(db_name=db_name, tool_queue=tool_queue)

    while tool_queue:
        # tools_str = "\n\n".join([str(tool) for tool in tool_queue])  # type: ignore
        # logger.info(f"\nTOOL QUEUE:\n\n{tools_str}\n")
        current_tool = await pop_next_tool(db_name=db_name)

        # if not isinstance(current_tool, Tool) and callable(current_tool):
        #     current_tool_name = (
        #         current_tool.__name__ if not isinstance(current_tool, partial) else current_tool.func.__name__
        #     )
        #     await update_current_tool(tool_id=current_tool.id, db_name=db_name)
        #     res = current_tool()
        #     if res:
        #         if isinstance(res, Dialog):
        #             dialog = res
        #         else:
        #             dialog = add_message(
        #                 dialog=dialog, message=assistant_message(content=str(res)), tag_messages=tag_messages
        #             )
        #     continue

        current_tool_name = current_tool.name
        await update_current_tool(tool_id=str(current_tool.id), db_name=db_name)
        dialog = await run_selected_tool(db_name=db_name, creator=creator, dialog=dialog, tool=current_tool)
        tool_queue = await get_tool_queue(db_name=db_name)
        await update_tool_queue(db_name=db_name, tool_queue=tool_queue)

    if continue_dialog:
        current_tool = await get_current_tool(db_name=db_name)
        current_tool_name = current_tool.name
        seq = int(current_tool_name == "ask_user")
        while True:
            if seq % 2 == 0 or current_tool_name == "call_ai":
                user_input = input("('q' or 'exit' or 'quit' to quit) > ")
                if user_input.lower() in ["q", "exit", "quit"]:
                    break
                dialog = await run_tools(
                    db_name=db_name,
                    creator=creator,
                    dialog=await new_task(
                        db_name=db_name, dialog=dialog, task=user_input, available_tools=available_tools
                    ),
                    continue_dialog=False,
                )
            else:
                current_tool_name = "call_ai"
                await update_current_tool(tool_id=str(current_tool.id), db_name=db_name)
                res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
                logger.success(f"ai response: {res}")
                dialog = add_message(
                    dialog=dialog, message=assistant_message(content=res), tag_messages=tag_messages
                )
            seq += 1

    logger.success(f"Messages: {dialog.messages[-2:]}")
    return dialog
