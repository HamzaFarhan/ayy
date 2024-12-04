import inspect
from functools import partial
from typing import Literal

from instructor import AsyncInstructor, Instructor
from loguru import logger
from pydantic import UUID4, BaseModel, create_model

from ayy import tools
from ayy.dialog import (
    Dialog,
    DialogToolSignature,
    MessagePurpose,
    ModelName,
    Tool,
    add_message,
    assistant_message,
    create_creator,
    dialog_to_kwargs,
    get_last_message,
    user_message,
)
from ayy.dialogs import DIALOG_NAMER_DIALOG
from ayy.func_utils import function_to_type, get_function_info, get_functions_from_module
from ayy.torm import (
    add_dialog_tools,
    get_dialog_tools,
    get_dialogs_with_signatures,
    get_next_dialog_tool,
    load_dialog,
    save_dialog,
    toggle_dialog_tool_usage,
)

MODEL_NAME = ModelName.GEMINI_FLASH
PINNED_TOOLS = set(["ask_user", "call_ai"])
CONTINUE_DIALOG = True
DEFAULT_PROMPT = "Generate a response if you've been asked. Otherwise, ask the user how they are doing."
DEFAULT_TOOL = Tool(reasoning="", name="call_ai", prompt=DEFAULT_PROMPT)


def get_dialog_signature(dialog: Dialog) -> DialogToolSignature | None:
    namer = create_creator(model_name=DIALOG_NAMER_DIALOG.model_name)
    namer_res = namer.create(
        **dialog_to_kwargs(dialog=DIALOG_NAMER_DIALOG, messages=dialog.messages),
        response_model=DialogToolSignature,
    )
    logger.info(f"dialog signature: {namer_res}")
    if namer_res.name == "":  # type: ignore
        return None
    return namer_res  # type: ignore


async def run_ask_user(dialog: Dialog, tool: Tool, db_name: str) -> Dialog:
    tool.prompt = tool.prompt or DEFAULT_PROMPT
    res = input(f"{tool.prompt}\n> ")
    dialog.messages += [assistant_message(content=tool.prompt), user_message(content=res)]
    await save_dialog(dialog=dialog, db_name=db_name)
    return dialog


async def run_call_ai(
    dialog: UUID4 | str | Dialog,
    tool: Tool,
    db_name: str,
    creator: Instructor | AsyncInstructor | None = None,
    memory_tagger_dialog: UUID4 | str | Dialog | None = None,
) -> Dialog:
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    if memory_tagger_dialog is not None:
        memory_tagger_dialog = await load_dialog(dialog=memory_tagger_dialog, db_name=db_name)
    creator = creator or create_creator(model_name=dialog.model_name)
    assistant_message_purpose = (
        MessagePurpose.CONVO if tool.name in PINNED_TOOLS | {"get_selected_tools"} else MessagePurpose.TOOL
    )
    tool.prompt = tool.prompt or DEFAULT_PROMPT
    logger.info(f"adding user message: {tool.prompt}")
    dialog = add_message(dialog=dialog, message=user_message(content=tool.prompt, purpose=MessagePurpose.TOOL))
    logger.info(f"\n\nCalling AI with messages: {dialog.messages}\n\n")
    res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
    logger.success(f"call_ai result: {res}")
    logger.info(f"adding assistant message: {res}")
    dialog = add_message(
        dialog=dialog,
        message=assistant_message(content=res, purpose=assistant_message_purpose),
        memory_tagger_dialog=memory_tagger_dialog,
    )
    await save_dialog(dialog=dialog, db_name=db_name)
    return dialog


async def get_selected_tools(db_name: str, dialog: UUID4 | str | Dialog, selected_tools: list[Tool]) -> Dialog:
    "Get and push a list of selected tools for the task"
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    tools_str = "\n".join([f"Tool {i}:\n{tool}" for i, tool in enumerate(selected_tools, start=1)])
    dialog = add_message(
        dialog=dialog, message=assistant_message(f"<selected_tools>\n{tools_str}\n</selected_tools>")
    )
    await save_dialog(dialog=dialog, db_name=db_name)
    await add_dialog_tools(dialog=dialog, tools=selected_tools, db_name=db_name)
    return dialog


async def run_dialog_as_tool(db_name: str, dialog: UUID4 | str | Dialog, task: str) -> Dialog:
    dialog = await new_task(db_name=db_name, dialog=dialog, task=task, continue_dialog=False)
    return dialog


class Task(BaseModel):
    task: str


async def run_tool(
    db_name: str,
    dialog: UUID4 | str | Dialog,
    tool: Tool,
    tool_is_dialog: bool = False,
    available_tools: list[str] | set[str] | None = None,
    creator: Instructor | AsyncInstructor | None = None,
    ignore_default_values: bool = False,
    skip_default_params: bool = False,
    memory_tagger_dialog: UUID4 | str | Dialog | None = None,
) -> Dialog:
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    if memory_tagger_dialog is not None:
        memory_tagger_dialog = await load_dialog(dialog=memory_tagger_dialog, db_name=db_name)
    is_async = False
    if tool_is_dialog:
        selected_tool = partial(run_dialog_as_tool, db_name, await load_dialog(dialog=tool.name, db_name=db_name))
        is_async = True
    else:
        try:
            selected_tool = getattr(tools, tool.name, globals().get(tool.name, None))
            if selected_tool is None:
                raise ValueError(f"Tool '{tool.name}' not found in tools module or current module")
            if not inspect.isfunction(selected_tool):
                raise ValueError(
                    f"Tool '{tool.name}' is not a function.\nGot {type(selected_tool).__name__} instead"
                )
            is_async = inspect.iscoroutinefunction(selected_tool)
        except AttributeError:
            raise ValueError(f"Tool '{tool.name}' not found in tools module")
    if tool.prompt:
        logger.info(f"adding user message: {tool.prompt}")
        message_purpose = MessagePurpose.CONVO if tool.name == "get_selected_tools" else MessagePurpose.TOOL
        dialog = add_message(dialog=dialog, message=user_message(content=tool.prompt, purpose=message_purpose))
        await save_dialog(dialog=dialog, db_name=db_name)
    tool_type = (
        getattr(tools, "tool_types", globals().get("tool_types", {})).get(tool.name, None)
        if not tool_is_dialog
        else Task
    )
    all_tools = available_tools or [tool_member[0] for tool_member in get_functions_from_module(module=tools)]
    if tool.name == "get_selected_tools" and len(all_tools) > 0:
        selected_tool = partial(get_selected_tools, db_name, dialog)
        tool_type = list[
            create_model(
                "SelectedTool",
                name=(Literal[*all_tools], ...),  # type: ignore
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
    # logger.info(f"\n\nCalling {tool.name} with tool_type: {tool_type}\n\n")
    creator = creator or create_creator(model_name=dialog.model_name)
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
        if res.name == dialog.name:
            return res
        res = get_last_message(messages=res.messages, role="assistant")
        if res is not None:
            res = res["content"]
    logger.info(f"adding assistant message: {res}")
    if res is not None:
        dialog = add_message(
            dialog=dialog, message=assistant_message(content=str(res)), memory_tagger_dialog=memory_tagger_dialog
        )
    await save_dialog(dialog=dialog, db_name=db_name)
    return dialog


async def run_selected_tool(
    db_name: str,
    dialog: UUID4 | str | Dialog,
    tool: Tool,
    tool_is_dialog: bool = False,
    available_tools: list[str] | set[str] | None = None,
    creator: Instructor | AsyncInstructor | None = None,
) -> Dialog:
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    creator = creator or create_creator(model_name=dialog.model_name)
    if tool.name.lower() == "ask_user":
        dialog = await run_ask_user(dialog=dialog, tool=tool, db_name=db_name)
    elif tool.name.lower() == "call_ai":
        dialog = await run_call_ai(creator=creator, dialog=dialog, tool=tool, db_name=db_name)
    else:
        dialog = await run_tool(
            db_name=db_name,
            creator=creator,
            dialog=dialog,
            tool=tool,
            tool_is_dialog=tool_is_dialog,
            available_tools=available_tools,
        )
    return dialog


async def run_tools(
    db_name: str,
    dialog: UUID4 | str | Dialog,
    creator: Instructor | AsyncInstructor | None = None,
    available_tools: list[str] | set[str] | None = None,
    dialogs: list[str] | set[str] | None = None,
    memory_tagger_dialog: UUID4 | str | Dialog | None = None,
    continue_dialog: bool = CONTINUE_DIALOG,
) -> Dialog:
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    creator = creator or create_creator(model_name=dialog.model_name)
    if memory_tagger_dialog is not None:
        memory_tagger_dialog = await load_dialog(dialog=memory_tagger_dialog, db_name=db_name)
    while True:
        next_tool = await get_next_dialog_tool(dialog=dialog, db_name=db_name)
        logger.info(f"next_tool: {next_tool}")
        if next_tool is None:
            break
        dialog = await run_selected_tool(
            db_name=db_name,
            creator=creator,
            dialog=dialog,
            tool=next_tool.tool,
            tool_is_dialog=next_tool.tool.name in dialogs if dialogs else False,
            available_tools=available_tools,
        )
        await toggle_dialog_tool_usage(dialog_tool_id=next_tool.id, db_name=db_name)
    used_tools = await get_dialog_tools(dialog=dialog, db_name=db_name, used=True)
    last_used_tool = used_tools[-1].tool if used_tools else DEFAULT_TOOL
    if continue_dialog:
        seq = int(last_used_tool.name == "ask_user")
        while True:
            if seq % 2 == 0 or last_used_tool.name == "call_ai":
                user_input = input("('q' or 'exit' or 'quit' to quit) > ")
                if user_input.lower().strip() in ["q", "exit", "quit"]:
                    return dialog
                dialog = await new_task(
                    db_name=db_name,
                    dialog=dialog,
                    task=user_input,
                    # available_tools=available_tools,
                    continue_dialog=False,
                )
            else:
                await add_dialog_tools(dialog=dialog, tools=DEFAULT_TOOL, db_name=db_name, used=True)
                res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
                logger.success(f"ai response: {res}")
                logger.info(f"adding assistant message: {res}")
                dialog = add_message(
                    dialog=dialog,
                    message=assistant_message(content=res),
                    memory_tagger_dialog=memory_tagger_dialog,
                )
                await save_dialog(dialog=dialog, db_name=db_name)
            seq += 1

    logger.success(f"Messages: {dialog.messages[-2:]}")
    dialog_signature = get_dialog_signature(dialog=dialog)
    await save_dialog(dialog=dialog, db_name=db_name, dialog_tool_signature=dialog_signature)
    return dialog


async def new_task(
    db_name: str,
    dialog: UUID4 | str | Dialog,
    task: str,
    creator: Instructor | AsyncInstructor | None = None,
    available_tools: list[str] | set[str] | None = None,
    memory_tagger_dialog: UUID4 | str | Dialog | None = None,
    continue_dialog: bool = CONTINUE_DIALOG,
) -> Dialog:
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    available_tools = (set(available_tools or []) | set(dialog.available_tools or [])) or []
    dialogs = await get_dialogs_with_signatures(db_name=db_name)
    dialog_names = []
    dialogs_as_tools = []
    for d in dialogs:
        if d.name != dialog.name:
            dialog_names.append(d.name)
            dialogs_as_tools.append(f"Tool:\n{d.dialog_tool_signature}")
    tool_names = []
    tools_list = []
    for _, func in get_functions_from_module(module=tools):
        if not available_tools or func.__name__ in set(available_tools) | PINNED_TOOLS:
            tool_names.append(func.__name__)
            tools_list.append(f"Tool:\n{get_function_info(func)}")
    tools_info = "\n\n".join(tools_list + dialogs_as_tools)
    dialog = add_message(
        dialog=dialog,
        message=user_message(content=f"Available tools for this task:\n{tools_info}", purpose=MessagePurpose.TOOL),
    )
    await save_dialog(dialog=dialog, db_name=db_name)
    await add_dialog_tools(
        dialog=dialog, tools=[Tool(reasoning="", name="get_selected_tools", prompt=task)], db_name=db_name
    )
    return await run_tools(
        db_name=db_name,
        dialog=dialog,
        creator=creator,
        available_tools=set(tool_names) | set(dialog_names) | PINNED_TOOLS,
        dialogs=set(dialog_names),
        memory_tagger_dialog=memory_tagger_dialog,
        continue_dialog=continue_dialog,
    )
