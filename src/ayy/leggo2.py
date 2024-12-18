import inspect
import json
from functools import partial
from importlib import import_module
from typing import Any, Callable, Literal

from instructor import AsyncInstructor, Instructor
from loguru import logger
from pydantic import UUID4, BaseModel, Field, create_model

from ayy.agent import (
    Agent,
    AgentToolSignature,
    ModelName,
    assistant_message,
    create_creator,
    agent_to_kwargs,
    get_last_message,
    user_message
)
from ayy.task import Task, TaskTool, Tool
from ayy.dialogs import DIALOG_NAMER_DIALOG, SUMMARIZER_DIALOG
from ayy.func_utils import function_to_type, get_function_info, get_functions_from_module
from ayy.prompts import MOVE_ON
from ayy.torm import (
    add_task_tools,
    get_dialogs_with_signatures,
    get_next_task_tool,
    get_task_tools,
    load_dialog,
    load_task,
    save_dialog,
    save_task,
    toggle_task_tool_usage,
)

MODEL_NAME = ModelName.GEMINI_FLASH
DEFAULT_TOOLS_MODULE = "ayy.tools"
PINNED_TOOLS = set(["ask_user", "call_ai"])
CONTINUE_DIALOG = True
DEFAULT_PROMPT = "Generate a response if you've been asked. Otherwise, ask the user how they are doing."
DEFAULT_TOOL = Tool(reasoning="", name="call_ai", prompt=DEFAULT_PROMPT)
TOOL_ATTEMPT_LIMIT = 3


class MoveOn(BaseModel):
    information_so_far: str
    move_on: bool = False
    next_assistant_task: str = Field(
        default="", description="The task from the user that the assistant should respond to or complete."
    )
    next_user_prompt: str = Field(
        default="", description="The prompt to ask the user for the remaining information."
    )


class TaskQuery(BaseModel):
    task_query: str


async def handle_creator_error(task: Task, db_name: str, error: Exception, tool_name: str = ""):
    res = f"Whoops! Something went wrong. Here's the error:\n{error}"
    if tool_name:
        res = f"Whoops! Something went wrong in '{tool_name}'. Here's the error:\n{error}"
    await add_task_tools(
        task=task, tools=[Tool(reasoning=res, name="ask_user", prompt=res)], db_name=db_name, run_next=True
    )


def get_dialog_signature(dialog: Dialog) -> DialogToolSignature | None:
    namer = create_creator(model_name=DIALOG_NAMER_DIALOG.model_name)
    try:
        namer_res = namer.create(
            **dialog_to_kwargs(dialog=DIALOG_NAMER_DIALOG, messages=dialog.messages),
            response_model=DialogToolSignature,
        )
        logger.info(f"dialog signature: {namer_res}")
    except Exception:
        logger.exception("Error getting dialog signature")
        return None
    return None if namer_res.name == "" else namer_res  # type: ignore


async def _continue_dialog(
    db_name: str,
    task: Task,
    dialog: Dialog,
    seq: int,
    last_used_tool: Tool,
    creator: Instructor | AsyncInstructor,
    available_tools: list[str] | set[str] | None = None,
    summarizer_dialog: Dialog | None = SUMMARIZER_DIALOG,
) -> tuple[Task, Dialog]:
    while True:
        if seq % 2 == 0 or last_used_tool.name == "call_ai":
            user_input = input("('q' or 'exit' or 'quit' to quit) > ")
            if user_input.lower().strip() in ["q", "exit", "quit"]:
                return task, dialog
            task, dialog = await new_task(
                db_name=db_name,
                dialog=dialog,
                task_query=user_input,
                available_tools=available_tools,
                continue_dialog=False,
                summarizer_dialog=summarizer_dialog,
            )
        else:
            await add_task_tools(task=task, tools=DEFAULT_TOOL, db_name=db_name, used=True)
            try:
                res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
                logger.success(f"ai response: {res}")
                res_message = assistant_message(content=res)
            except Exception as e:
                logger.exception("Error calling AI after continuing dialog")
                res_message = assistant_message(
                    content=f"Whoops! Something went wrong. Here's the error:\n{e}",
                    purpose=MessagePurpose.ERROR,
                )
            logger.info(f"adding message: {res_message['content']}")
            task = add_task_message(task=task, message=res_message)
            dialog = add_dialog_message(dialog=dialog, message=res_message, summarizer_dialog=summarizer_dialog)
            await save_dialog(dialog=dialog, db_name=db_name)
            await save_task(task=task, db_name=db_name)
        seq += 1


async def _run_ask_user(
    db_name: str,
    task: UUID4 | str | Task,
    dialog: UUID4 | str | Dialog,
    task_query: str = "",
    summarizer_dialog: Dialog | None = SUMMARIZER_DIALOG,
) -> tuple[Task, Dialog]:
    task = await load_task(task=task, db_name=db_name)
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    task_query = task_query or DEFAULT_PROMPT
    res = input(f"{task_query}\n> ")
    task = add_task_message(task=task, message=assistant_message(content=task_query))
    task = add_task_message(task=task, message=user_message(content=res))
    dialog = add_dialog_message(
        dialog=dialog, message=assistant_message(content=task_query), summarizer_dialog=summarizer_dialog
    )
    dialog = add_dialog_message(
        dialog=dialog, message=user_message(content=res), summarizer_dialog=summarizer_dialog
    )
    await save_dialog(dialog=dialog, db_name=db_name)
    await save_task(task=task, db_name=db_name)
    return task, dialog


async def run_ask_user(
    db_name: str,
    task: UUID4 | str | Task,
    dialog: UUID4 | str | Dialog,
    task_query: str = "",
    creator: Instructor | AsyncInstructor | None = None,
    available_tools: list[str] | set[str] | None = None,
    summarizer_dialog: Dialog | None = SUMMARIZER_DIALOG,
) -> tuple[Task, Dialog]:
    task = await load_task(task=task, db_name=db_name)
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    creator = creator or create_creator(model_name=dialog.model_name)
    move_on = MoveOn(information_so_far="", next_user_prompt=task_query)
    while not move_on.move_on:
        if not move_on.next_user_prompt and not move_on.next_assistant_task:
            break
        if move_on.next_assistant_task:
            task, dialog = await new_task(
                db_name=db_name,
                dialog=dialog,
                task_query=move_on.next_assistant_task,
                available_tools=available_tools,
                continue_dialog=False,
                summarizer_dialog=summarizer_dialog,
            )
        else:
            task, dialog = await _run_ask_user(
                db_name=db_name,
                task=task,
                dialog=dialog,
                task_query=move_on.next_user_prompt,
                summarizer_dialog=summarizer_dialog,
            )
        try:
            move_on: MoveOn = creator.create(
                **dialog_to_kwargs(
                    dialog=dialog,
                    messages=[user_message(content=MOVE_ON)],
                ),
                response_model=MoveOn,
            )  # type: ignore
            logger.info(f"move_on: {move_on}")
        except Exception:
            logger.exception("Error getting move_on")
            move_on = MoveOn(information_so_far="", move_on=True, next_user_prompt="")
    if move_on.information_so_far:
        info_message = assistant_message(
            content=f"<information_so_far>\n{move_on.information_so_far}\n</information_so_far>",
            purpose=MessagePurpose.TOOL,
        )
        dialog = add_dialog_message(dialog=dialog, message=info_message)
        await save_dialog(dialog=dialog, db_name=db_name)
    return task, dialog


async def run_call_ai(
    db_name: str,
    task: UUID4 | str | Task,
    dialog: UUID4 | str | Dialog,
    tool: Tool,
    creator: Instructor | AsyncInstructor | None = None,
    summarizer_dialog: Dialog | None = SUMMARIZER_DIALOG,
) -> tuple[Task, Dialog]:
    task = await load_task(task=task, db_name=db_name)
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    creator = creator or create_creator(model_name=dialog.model_name)
    assistant_message_purpose = (
        MessagePurpose.CONVO if tool.name in PINNED_TOOLS | {"get_selected_tools"} else MessagePurpose.TOOL
    )
    tool.prompt = tool.prompt or DEFAULT_PROMPT
    logger.info(f"adding user message: {tool.prompt}")
    tp_message = user_message(content=tool.prompt, purpose=MessagePurpose.TOOL)
    # task = add_task_message(task=task, message=tp_message)
    dialog = add_dialog_message(dialog=dialog, message=tp_message, summarizer_dialog=summarizer_dialog)
    logger.info(f"\n\nCalling AI with messages: {task.messages}\n\n")
    res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
    logger.success(f"call_ai result: {res}")
    logger.info(f"adding assistant message: {res}")
    res_message = assistant_message(content=res, purpose=assistant_message_purpose)
    if assistant_message_purpose != MessagePurpose.TOOL:
        task = add_task_message(task=task, message=res_message)
    dialog = add_dialog_message(dialog=dialog, message=res_message, summarizer_dialog=summarizer_dialog)
    await save_dialog(dialog=dialog, db_name=db_name)
    await save_task(task=task, db_name=db_name)
    return task, dialog


async def get_selected_tools(
    db_name: str, task: UUID4 | str | Task, dialog: UUID4 | str | Dialog, selected_tools: list[Tool]
) -> tuple[Task, Dialog]:
    "Get and push a list of selected tools for the task"
    task = await load_task(task=task, db_name=db_name)
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    tools_str = "\n".join([f"Tool {i}:\n{tool}" for i, tool in enumerate(selected_tools, start=1)])
    st_message = assistant_message(
        f"<selected_tools>\n{tools_str}\n</selected_tools>", purpose=MessagePurpose.TOOL
    )
    # task = add_task_message(task=task, message=st_message)
    dialog = add_dialog_message(dialog=dialog, message=st_message)
    await save_dialog(dialog=dialog, db_name=db_name)
    await save_task(task=task, db_name=db_name)
    await add_task_tools(task=task, tools=selected_tools, db_name=db_name)  # type: ignore
    return task, dialog


async def run_dialog_as_tool(
    db_name: str, dialog: UUID4 | str | Dialog, available_tools: list[str] | set[str] | None, task_query: str
) -> Task:
    task, _ = await new_task(
        db_name=db_name,
        dialog=dialog,
        task_query=task_query,
        continue_dialog=False,
        available_tools=available_tools,
    )
    return task


async def _run_selected_tool(selected_tool: Callable, tool_args: Any, is_async: bool = False) -> Any:
    if tool_args is None:
        return selected_tool()
    if isinstance(tool_args, BaseModel):
        if is_async:
            return await selected_tool(**tool_args.model_dump())
        else:
            return selected_tool(**tool_args.model_dump())
    elif is_async:
        return await selected_tool(tool_args)
    else:
        return selected_tool(tool_args)


async def run_tool(
    db_name: str,
    task: UUID4 | str | Task,
    dialog: UUID4 | str | Dialog,
    task_tool: TaskTool,
    tool_is_dialog: bool = False,
    creator: Instructor | AsyncInstructor | None = None,
    available_tools: list[str] | set[str] | None = None,
    ignore_default_values: bool = False,
    skip_default_params: bool = False,
    summarizer_dialog: Dialog | None = SUMMARIZER_DIALOG,
    tools_module: str = DEFAULT_TOOLS_MODULE,
) -> tuple[Task, Dialog]:
    task = await load_task(task=task, db_name=db_name)
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    tool = task_tool.tool
    tools = import_module(tools_module)
    is_async = False
    if tool_is_dialog:
        selected_tool = partial(
            run_dialog_as_tool, db_name, await load_dialog(dialog=tool.name, db_name=db_name), available_tools
        )
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
    # if tool.prompt:
    #     logger.info(f"adding user message: {tool.prompt}")
    #     message_purpose = MessagePurpose.CONVO if tool.name == "get_selected_tools" else MessagePurpose.TOOL
    #     tp_message = user_message(content=tool.prompt, purpose=message_purpose)
    #     if message_purpose != MessagePurpose.TOOL:
    #         task = add_task_message(task=task, message=tp_message)
    #     dialog = add_dialog_message(dialog=dialog, message=tp_message, summarizer_dialog=summarizer_dialog)
    #     await save_dialog(dialog=dialog, db_name=db_name)
    #     await save_task(task=task, db_name=db_name)
    tool_type = (
        getattr(tools, "tool_types", globals().get("tool_types", {})).get(tool.name, None)
        if not tool_is_dialog
        else TaskQuery
    )
    all_tools = available_tools or [tool_member[0] for tool_member in get_functions_from_module(module=tools)]
    if tool.name == "get_selected_tools" and len(all_tools) > 0:
        selected_tool = partial(get_selected_tools, db_name, task, dialog)
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
    if tool_type is None:
        tool_args = None
    # logger.info(f"\n\nCalling {tool.name} with tool_type: {tool_type}\n\n")
    creator = creator or create_creator(model_name=dialog.model_name)
    tool_args = creator.create(
        **dialog_to_kwargs(dialog=dialog),
        response_model=tool_type,  # type: ignore
    )
    logger.info(f"{tool.name} tool_args: {tool_args}")
    if tool.name != "get_selected_tools":
        try:
            cr_message = assistant_message(
                content=f"Generated for {tool.name}: {tool_args}", purpose=MessagePurpose.TOOL
            )
            task_tool.args_message = cr_message
            await add_task_tools(task=task, tools=task_tool, db_name=db_name)
            # dialog = add_dialog_message(dialog=dialog, message=cr_message, summarizer_dialog=summarizer_dialog)
            # await save_dialog(dialog=dialog, db_name=db_name)
        except Exception:
            pass
    res = await _run_selected_tool(selected_tool=selected_tool, tool_args=tool_args, is_async=is_async)
    if isinstance(res, tuple) and isinstance(res[0], Task) and isinstance(res[1], Dialog):
        res = res[0]
    logger.success(f"{tool.name} result: {res}")
    if isinstance(res, Task):
        if res.name in [dialog.name, task.name]:
            return res, dialog
        res = get_last_message(messages=res.messages, role="assistant")
        if res is not None:
            res = res["content"]
    if res is not None:
        logger.info(f"adding assistant message: {res}")
        res_message = assistant_message(content=str(res), purpose=MessagePurpose.TOOL)
        task_tool.result_message = res_message
        await add_task_tools(task=task, tools=task_tool, db_name=db_name)
        # task = add_task_message(task=task, message=res_message)
        # dialog = add_dialog_message(dialog=dialog, message=res_message, summarizer_dialog=summarizer_dialog)
    await save_dialog(dialog=dialog, db_name=db_name)
    await save_task(task=task, db_name=db_name)
    return task, dialog


async def run_selected_tool(
    db_name: str,
    task: UUID4 | str | Task,
    dialog: UUID4 | str | Dialog,
    task_tool: TaskTool,
    tool_is_dialog: bool = False,
    creator: Instructor | AsyncInstructor | None = None,
    available_tools: list[str] | set[str] | None = None,
    summarizer_dialog: Dialog | None = SUMMARIZER_DIALOG,
    tools_module: str = DEFAULT_TOOLS_MODULE,
) -> tuple[Task, Dialog]:
    task = await load_task(task=task, db_name=db_name)
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    creator = creator or create_creator(model_name=dialog.model_name)
    if task_tool.tool.name.lower() == "ask_user":
        task, dialog = await run_ask_user(
            db_name=db_name,
            task=task,
            dialog=dialog,
            task_query=task_tool.tool.prompt,
            summarizer_dialog=summarizer_dialog,
        )
    elif task_tool.tool.name.lower() == "call_ai":
        task, dialog = await run_call_ai(
            db_name=db_name,
            task=task,
            dialog=dialog,
            tool=task_tool.tool,
            creator=creator,
            summarizer_dialog=summarizer_dialog,
        )
    else:
        task, dialog = await run_tool(
            db_name=db_name,
            task=task,
            dialog=dialog,
            task_tool=task_tool,
            tool_is_dialog=tool_is_dialog,
            available_tools=available_tools,
            creator=creator,
            summarizer_dialog=summarizer_dialog,
            tools_module=tools_module,
        )
    return task, dialog


async def run_tools(
    db_name: str,
    task: UUID4 | str | Task,
    dialog: UUID4 | str | Dialog,
    creator: Instructor | AsyncInstructor | None = None,
    available_tools: list[str] | set[str] | None = None,
    dialogs: list[str] | set[str] | None = None,
    continue_dialog: bool = CONTINUE_DIALOG,
    summarizer_dialog: Dialog | None = SUMMARIZER_DIALOG,
    tools_module: str = DEFAULT_TOOLS_MODULE,
) -> tuple[Task, Dialog]:
    task = await load_task(task=task, db_name=db_name)
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    creator = creator or create_creator(model_name=dialog.model_name)
    while True:
        next_tool = await get_next_task_tool(task=task, db_name=db_name)
        logger.info(f"next_tool: {next_tool}")
        if next_tool is None:
            break
        attempts = 0
        while attempts < TOOL_ATTEMPT_LIMIT:
            try:
                task, dialog = await run_selected_tool(
                    db_name=db_name,
                    task=task,
                    creator=creator,
                    dialog=dialog,
                    task_tool=next_tool,
                    tool_is_dialog=next_tool.tool.name in dialogs if dialogs else False,
                    available_tools=available_tools,
                    summarizer_dialog=summarizer_dialog,
                    tools_module=tools_module,
                )
                await toggle_task_tool_usage(task_tool_id=next_tool.id, db_name=db_name)
                break
            except Exception as e:
                logger.exception(f"Error running tool '{next_tool.tool.name}'")
                await handle_creator_error(task=task, db_name=db_name, error=e, tool_name=next_tool.tool.name)
                attempts += 1
    used_tools = await get_task_tools(task=task, db_name=db_name, used=True)
    last_used_tool = used_tools[-1].tool if used_tools else DEFAULT_TOOL
    if continue_dialog:
        task, dialog = await _continue_dialog(
            db_name=db_name,
            task=task,
            dialog=dialog,
            seq=int(last_used_tool.name == "ask_user"),
            last_used_tool=last_used_tool,
            creator=creator,
            available_tools=available_tools,
            summarizer_dialog=summarizer_dialog,
        )

    logger.success(f"Messages: {task.messages[-2:]}")
    dialog_signature = (
        DialogToolSignature(**dialog.dialog_tool_signature)
        if dialog.dialog_tool_signature
        else get_dialog_signature(dialog=dialog)
    )
    await save_dialog(dialog=dialog, db_name=db_name, dialog_tool_signature=dialog_signature)
    await save_task(task=task, db_name=db_name)
    return task, dialog


async def new_task(
    db_name: str,
    dialog: UUID4 | str | Dialog,
    task_query: str,
    task_name: str = "",
    task: Task | None = None,
    creator: Instructor | AsyncInstructor | None = None,
    available_tools: list[str] | set[str] | None = None,
    recommended_tools: dict[int, str] | list[str] | None = None,
    selected_tools: list[Tool] | None = None,
    continue_dialog: bool = CONTINUE_DIALOG,
    summarizer_dialog: Dialog | None = SUMMARIZER_DIALOG,
    tools_module: str = DEFAULT_TOOLS_MODULE,
) -> tuple[Task, Dialog]:
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    task = task or Task(name=task_name, dialog_id=dialog.id)
    tools = import_module(tools_module)
    dialogs = await get_dialogs_with_signatures(db_name=db_name)
    dialog_names = []
    dialogs_as_tools = []
    for d in dialogs:
        if d.name != dialog.name:
            dialog_names.append(d.name)
            dialogs_as_tools.append(f"Tool:\n{d.dialog_tool_signature}")
    tool_names = []
    tools_list = []
    available_tools = set(available_tools or dialog.available_tools or [])
    selected_tools = selected_tools or []
    for _, func in get_functions_from_module(module=tools):
        if not available_tools or func.__name__ in available_tools | PINNED_TOOLS | set(
            tool.name for tool in selected_tools
        ):
            tool_names.append(func.__name__)
            tools_list.append(f"Tool:\n{get_function_info(func)}")
    tools_info = "\n\n".join(tools_list + dialogs_as_tools)

    av_message = user_message(
        content=f"<available_tools_for_task>\n{tools_info}\n</available_tools_for_task>",
        purpose=MessagePurpose.AVAILABLE_TOOLS,
    )
    task.available_tools_message = av_message
    # dialog = add_dialog_message(dialog=dialog, message=av_message, summarizer_dialog=summarizer_dialog)
    if recommended_tools is None:
        used_tools = await get_task_tools(task=task, db_name=db_name, used=True)
        recommended_tools = {tool.position: tool.tool.name for tool in used_tools}
    if recommended_tools:
        rec_message = user_message(
            content=f"<recommended_tools_for_task_in_order>\n{json.dumps(recommended_tools, indent=2)}\n</recommended_tools_for_task_in_order>",
            purpose=MessagePurpose.RECOMMENDED_TOOLS,
        )
        task.recommended_tools_message = rec_message
        # dialog = add_dialog_message(dialog=dialog, message=rec_message, summarizer_dialog=summarizer_dialog)
    # await save_dialog(dialog=dialog, db_name=db_name)
    await save_task(task=task, db_name=db_name)
    await add_task_tools(
        task=task,
        tools=selected_tools or [Tool(reasoning="", name="get_selected_tools", prompt=task_query)],  # type: ignore
        db_name=db_name,
    )
    return await run_tools(
        db_name=db_name,
        task=task,
        dialog=dialog,
        creator=creator,
        available_tools=set(tool_names) | set(dialog_names) | PINNED_TOOLS,
        dialogs=set(dialog_names),
        continue_dialog=continue_dialog,
        summarizer_dialog=summarizer_dialog,
        tools_module=tools_module,
    )
