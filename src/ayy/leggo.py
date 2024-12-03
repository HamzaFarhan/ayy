import inspect
from functools import partial
from typing import Literal
from uuid import uuid4

from instructor import AsyncInstructor, Instructor
from loguru import logger
from pydantic import UUID4, BaseModel, create_model

from ayy import tools
from ayy.dialog import (
    DEFAULT_PROMPT,
    DEFAULT_TOOL,
    Dialog,
    MessagePurpose,
    ModelName,
    Tool,
    add_message,
    assistant_message,
    create_creator,
    dialog_to_kwargs,
    user_message,
)
from ayy.func_utils import function_to_type, get_function_info, get_functions_from_module
from ayy.torm import (
    add_dialog_tools,
    get_dialog_tools,
    get_next_dialog_tool,
    load_dialog,
    save_dialog,
    toggle_dialog_tool_usage,
)

MODEL_NAME = ModelName.GEMINI_FLASH
PINNED_TOOLS = set(["ask_user", "call_ai"])
CONTINUE_DIALOG = True


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


async def get_selected_tools(db_name: str, dialog: UUID4 | str | Dialog, selected_tools: list[Tool]):
    """
    Get and push a list of selected tools for the task
    It will also add an ask_user at the start to approve the tools. You don't need to add it yourself.
    """
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    await add_dialog_tools(dialog=dialog, tools=selected_tools, db_name=db_name)


async def run_tool(
    db_name: str,
    dialog: UUID4 | str | Dialog,
    tool: Tool,
    creator: Instructor | AsyncInstructor | None = None,
    ignore_default_values: bool = False,
    skip_default_params: bool = False,
    memory_tagger_dialog: UUID4 | str | Dialog | None = None,
) -> Dialog:
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    if memory_tagger_dialog is not None:
        memory_tagger_dialog = await load_dialog(dialog=memory_tagger_dialog, db_name=db_name)
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
        logger.info(f"adding user message: {tool.prompt}")
        message_purpose = MessagePurpose.CONVO if tool.name == "get_selected_tools" else MessagePurpose.TOOL
        dialog = add_message(dialog=dialog, message=user_message(content=tool.prompt, purpose=message_purpose))
        await save_dialog(dialog=dialog, db_name=db_name)

    tool_type = getattr(tools, "tool_types", globals().get("tool_types", {})).get(tool.name, None)
    all_tools = get_functions_from_module(module=tools)
    if tool.name == "get_selected_tools" and len(all_tools) > 0:
        selected_tool = partial(get_selected_tools, db_name, dialog)
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
        return res
    logger.info(f"adding assistant message: {res}")
    if res is not None:
        dialog = add_message(
            dialog=dialog, message=assistant_message(content=str(res)), memory_tagger_dialog=memory_tagger_dialog
        )
    await save_dialog(dialog=dialog, db_name=db_name)
    return dialog


async def run_selected_tool(
    db_name: str, dialog: UUID4 | str | Dialog, tool: Tool, creator: Instructor | AsyncInstructor | None = None
) -> Dialog:
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    creator = creator or create_creator(model_name=dialog.model_name)
    if tool.name.lower() == "ask_user":
        dialog = await run_ask_user(dialog=dialog, tool=tool, db_name=db_name)
    elif tool.name.lower() == "call_ai":
        dialog = await run_call_ai(creator=creator, dialog=dialog, tool=tool, db_name=db_name)
    else:
        dialog = await run_tool(db_name=db_name, creator=creator, dialog=dialog, tool=tool)
    return dialog


async def run_tools(
    db_name: str,
    dialog: UUID4 | str | Dialog,
    creator: Instructor | AsyncInstructor | None = None,
    available_tools: list[str] | set[str] | None = None,
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
        dialog = await run_selected_tool(db_name=db_name, creator=creator, dialog=dialog, tool=next_tool.tool)
        await toggle_dialog_tool_usage(dialog_tool_id=next_tool.id, db_name=db_name)
    used_tools = await get_dialog_tools(dialog=dialog, db_name=db_name, used=True)
    last_used_tool = used_tools[-1].tool if used_tools else DEFAULT_TOOL
    if continue_dialog:
        seq = int(last_used_tool.name == "ask_user")
        while True:
            if seq % 2 == 0 or last_used_tool.name == "call_ai":
                user_input = input("('q' or 'exit' or 'quit' to quit) > ")
                if user_input.lower() in ["q", "exit", "quit"]:
                    return dialog
                dialog = await run_tools(
                    db_name=db_name,
                    creator=creator,
                    dialog=await new_task(
                        db_name=db_name, dialog=dialog, task=user_input, available_tools=available_tools
                    ),
                    continue_dialog=False,
                )
            else:
                await add_dialog_tools(dialog=dialog, tools=[DEFAULT_TOOL], db_name=db_name)
                res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
                logger.success(f"ai response: {res}")
                logger.info(f"In run_tools, adding assistant message: {res}")
                dialog = add_message(
                    dialog=dialog,
                    message=assistant_message(content=res),
                    memory_tagger_dialog=memory_tagger_dialog,
                )
                await save_dialog(dialog=dialog, db_name=db_name)
            seq += 1

    logger.success(f"Messages: {dialog.messages[-2:]}")
    return dialog


async def new_task(
    db_name: str,
    dialog: UUID4 | str | Dialog,
    task: str,
    task_name: str = "",
    creator: Instructor | AsyncInstructor | None = None,
    available_tools: list[str] | set[str] | None = None,
    memory_tagger_dialog: UUID4 | str | Dialog | None = None,
    continue_dialog: bool = CONTINUE_DIALOG,
) -> Dialog:
    dialog = await load_dialog(dialog=dialog, db_name=db_name)
    if task_name and dialog.name != task_name:
        dialog.id = uuid4()
        dialog.name = task_name
    available_tools = available_tools or []
    tools_info = "\n\n".join(
        [
            f"Tool {i}:\n{get_function_info(func)}"
            for i, (_, func) in enumerate(get_functions_from_module(module=tools), start=1)
            if not available_tools or func.__name__ in set(available_tools) | PINNED_TOOLS
        ]
    )
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
        available_tools=available_tools,
        memory_tagger_dialog=memory_tagger_dialog,
        continue_dialog=continue_dialog,
    )


"""
So for tool selection, we give the AI a list of function signatures and it gives us a list or strings which are the names of the functions.
We then get the AI to generate the arguments for the functions. Now, the arguments needed are ifered from the function params. But we can also define some special "tool_type" for that function and trick the AI.
So for example if a tool has a signature like this:
def get_weather(day: Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], location: str) -> str:
    ...
But IF we have an entry in tool_types like this:
tool_types = {
    "get_weather": list[tuple[Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], str]]
}
Now, when the AI is generating the arguments, it will see that the tool_type for get_weather is a list of tuples with a string and a Literal type.
So it will generate a list of tuples with a string and a value from the Literal type, it won't generate just 1 arg for day and 1 for location. So even tho the selection was made based on the function signature, it will still generate a list of tuples because we have tricked it. This can come in handy at times.

Now, suppose I have 2 tools:
1. lookup_flights(day: str, location: str) -> str
2. book_flight(flight_number: str) -> str
3. check_match_score(team1: str, team2: str) -> str

And my task was "book a flight for tomorrow to london".
I call new_task with this task and it will create a dialog.
The AI will select tools 1 and 2 to be run in that order.
Once the task is finished, I name the dialog "lookup_flights_and_book" and save it.
This dialog has the whole conversation in it. Along with the tool calls and their results, and any hiccups along the way.
Whenever I get a similar task, I want the AI to send this task to this specific dialog.
One way to do that is to trick the AI into thinking that "lookup_flights_and_book" is also a tool. So it would just select one tool from the list of tools and call it.
I want a generic way to do this.
So, once a dialog is saved, I want it to become available as a tool.

"""
