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
    get_last_message,
    user_message,
)
from ayy.agents import AGENT_NAMER_AGENT
from ayy.func_utils import function_to_type, get_function_info, get_functions_from_module
from ayy.prompts import MOVE_ON
from ayy.task import Task, TaskTool, Tool, task_to_kwargs
from ayy.torm import (
    add_task_tool_args_messages,
    add_task_tool_result_messages,
    add_task_tools,
    get_agents_with_signatures,
    get_next_available_task_tool_id_and_position,
    get_next_task_tool,
    get_task_messages,
    get_task_tools,
    load_agent,
    save_agent,
    save_task,
    set_task_tool_used,
)

SUMMARIZER_AGENT = None
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


async def handle_creator_error(db_name: str, task: Task, error: Exception, tool_name: str = ""):
    res = f"Whoops! Something went wrong. Here's the error:\n{error}"
    if tool_name:
        res = f"Whoops! Something went wrong in '{tool_name}'. Here's the error:\n{error}"
    await add_task_tools(
        task=task, tools=[Tool(reasoning=res, name="ask_user", prompt=res)], db_name=db_name, run_next=True
    )


async def get_agent_tool_signature(
    db_name: str, task: Task, agent_namer_agent: Agent = AGENT_NAMER_AGENT
) -> AgentToolSignature | None:
    namer = create_creator(model_name=agent_namer_agent.model_name)
    try:
        task_tools = await get_task_tools(task=task, db_name=db_name, used=True)
        namer_res = namer.create(
            **task_to_kwargs(task=task, task_tools=task_tools, agent=agent_namer_agent),
            response_model=AgentToolSignature,
        )
        logger.info(f"agent signature: {namer_res}")
    except Exception:
        logger.exception("Error getting dialog signature")
        return None
    return None if namer_res.name == "" else namer_res  # type: ignore


async def _continue_dialog(
    db_name: str,
    task: Task,
    agent: Agent,
    seq: int,
    last_used_tool: Tool,
    creator: Instructor | AsyncInstructor,
    available_tools: list[str] | set[str] | None = None,
    summarizer_agent: Agent | None = SUMMARIZER_AGENT,
) -> Task:
    while True:
        if seq % 2 == 0 or last_used_tool.name == "call_ai":
            user_input = input("('q' or 'exit' or 'quit' to quit) > ")
            if user_input.lower().strip() in ["q", "exit", "quit"]:
                return task
            task = await new_task(
                db_name=db_name,
                agent=agent,
                task_query=user_input,
                available_tools=available_tools,
                continue_dialog=False,
                summarizer_agent=summarizer_agent,
            )
        else:
            (
                next_available_task_tool_id,
                next_available_task_tool_position,
            ) = await get_next_available_task_tool_id_and_position(task=task, db_name=db_name)
            ai_task_tool = TaskTool(
                id=next_available_task_tool_id,
                task_id=task.id,
                position=next_available_task_tool_position,
                tool=DEFAULT_TOOL,
            )
            await add_task_tools(task=task, tools=[ai_task_tool], db_name=db_name, run_next=True)
            try:
                task_tools = await get_task_tools(
                    task=task, db_name=db_name, used=None, max_position=next_available_task_tool_position
                )
                res = creator.create(
                    **task_to_kwargs(
                        task=task, task_tools=task_tools, agent=agent, summarizer_agent=summarizer_agent
                    ),
                    response_model=str,
                )
                logger.success(f"ai response: {res}")
                res_message = assistant_message(content=res)
            except Exception as e:
                logger.exception("Error calling AI after continuing dialog")
                res_message = assistant_message(content=f"Whoops! Something went wrong. Here's the error:\n{e}")
            logger.info(f"adding message: {res_message['content']}")
            ai_task_tool.tool_result_messages += [res_message]
            await add_task_tool_result_messages(
                task_tool_id=ai_task_tool.id, tool_result_messages=[res_message], db_name=db_name
            )
            # await save_task(task=task, db_name=db_name)
        seq += 1


async def _run_ask_user(db_name: str, task_tool: TaskTool) -> TaskTool:
    task_query = task_tool.tool.prompt or DEFAULT_PROMPT
    res = input(f"{task_query}\n> ")
    tool_result_message = user_message(content=res)
    task_tool.tool_result_messages += [tool_result_message]
    await add_task_tool_result_messages(
        task_tool_id=task_tool.id, tool_result_messages=[tool_result_message], db_name=db_name
    )
    return task_tool


async def run_ask_user(
    db_name: str,
    task: Task,
    agent: Agent,
    task_tool: TaskTool,
    creator: Instructor | AsyncInstructor | None = None,
    available_tools: list[str] | set[str] | None = None,
    summarizer_agent: Agent | None = SUMMARIZER_AGENT,
    max_tangent_messages: int = 10,
) -> Task:
    # task = await load_task(task=task, db_name=db_name)
    # agent = await load_agent(agent=agent, db_name=db_name, current_task_id=task.id)
    creator = creator or create_creator(model_name=agent.model_name)
    move_on = MoveOn(information_so_far="", next_user_prompt=task_tool.tool.prompt)
    tangent_task = None
    while not move_on.move_on:
        if not move_on.next_user_prompt and not move_on.next_assistant_task:
            break
        if move_on.next_assistant_task:
            tangent_task = await new_task(
                db_name=db_name,
                agent=agent,
                task_query=move_on.next_assistant_task,
                creator=creator,
                available_tools=available_tools,
                continue_dialog=False,
                summarizer_agent=summarizer_agent,
            )
        else:
            task_tool = await _run_ask_user(db_name=db_name, task_tool=task_tool)
        try:
            messages = [user_message(content=MOVE_ON)]
            if tangent_task is not None:
                tangent_task_messages = await get_task_messages(task=tangent_task, db_name=db_name, used=True)
                messages = tangent_task_messages[-max_tangent_messages:] + messages
            task_tools = await get_task_tools(
                task=task, db_name=db_name, used=None, max_position=task_tool.position
            )
            move_on: MoveOn = creator.create(
                **task_to_kwargs(
                    task=task,
                    task_tools=task_tools,
                    agent=agent,
                    messages=messages,
                    summarizer_agent=summarizer_agent,
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
        )
        task_tool.tool_result_messages += [info_message]
        await add_task_tool_result_messages(
            task_tool_id=task_tool.id, tool_result_messages=[info_message], db_name=db_name
        )
    # await save_task(task=task, db_name=db_name)
    return task


async def run_call_ai(
    db_name: str,
    task: Task,
    agent: Agent,
    task_tool: TaskTool,
    creator: Instructor | AsyncInstructor | None = None,
    summarizer_agent: Agent | None = SUMMARIZER_AGENT,
) -> Task:
    # task = await load_task(task=task, db_name=db_name)
    # agent = await load_agent(agent=agent, db_name=db_name, current_task_id=task.id)
    creator = creator or create_creator(model_name=agent.model_name)
    task_tool.tool.prompt = task_tool.tool.prompt or DEFAULT_PROMPT
    task_tools = await get_task_tools(task=task, db_name=db_name, used=None, max_position=task_tool.position)
    res = creator.create(
        **task_to_kwargs(task=task, task_tools=task_tools, agent=agent, summarizer_agent=summarizer_agent),
        response_model=str,
    )
    logger.success(f"call_ai result: {res}")
    logger.info(f"adding assistant message: {res}")
    res_message = assistant_message(content=res)
    task_tool.tool_result_messages += [res_message]
    await add_task_tool_result_messages(
        task_tool_id=task_tool.id, tool_result_messages=[res_message], db_name=db_name
    )
    # await save_agent(agent=agent, db_name=db_name)
    # await save_task(task=task, db_name=db_name)
    return task


async def get_selected_tools(db_name: str, task: Task, selected_tools: list[Tool]) -> Task:
    "Get and push a list of selected tools for the task"
    # task = await load_task(task=task, db_name=db_name)
    tools_str = "\n".join([f"Tool {i}:\n{tool}" for i, tool in enumerate(selected_tools, start=1)])
    st_message = assistant_message(f"<selected_tools>\n{tools_str}\n</selected_tools>")
    task.selected_tools_message = st_message
    # await save_task(task=task, db_name=db_name)
    await add_task_tools(task=task, tools=selected_tools, db_name=db_name)  # type: ignore
    return task


async def run_agent_as_tool(
    db_name: str,
    agent: Agent,
    available_tools: list[str] | set[str] | None,
    task_query: str,
    tools_module: str = DEFAULT_TOOLS_MODULE,
) -> Task:
    task = await new_task(
        db_name=db_name,
        agent=agent,
        task_query=task_query,
        continue_dialog=False,
        available_tools=available_tools,
        tools_module=tools_module,
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
    task: Task,
    agent: Agent,
    task_tool: TaskTool,
    tool_is_agent: bool = False,
    creator: Instructor | AsyncInstructor | None = None,
    available_tools: list[str] | set[str] | None = None,
    ignore_default_values: bool = False,
    skip_default_params: bool = False,
    summarizer_agent: Agent | None = SUMMARIZER_AGENT,
    tools_module: str = DEFAULT_TOOLS_MODULE,
) -> Task:
    # task = await load_task(task=task, db_name=db_name)
    # agent = await load_agent(agent=agent, db_name=db_name, current_task_id=task.id)
    tool = task_tool.tool
    tools = import_module(tools_module)
    is_async = False
    if tool_is_agent:
        selected_tool = partial(
            run_agent_as_tool,
            db_name,
            await load_agent(agent=tool.name, db_name=db_name, current_task_id=task.id),
            available_tools,
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
    tool_type = (
        getattr(tools, "tool_types", globals().get("tool_types", {})).get(tool.name, None)
        if not tool_is_agent
        else TaskQuery
    )
    all_tools = available_tools or [tool_member[0] for tool_member in get_functions_from_module(module=tools)]
    if tool.name == "get_selected_tools" and len(all_tools) > 0:
        selected_tool = partial(get_selected_tools, db_name, task)
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
    task_tools = await get_task_tools(task=task, db_name=db_name, used=None, max_position=task_tool.position)
    creator = creator or create_creator(model_name=agent.model_name)
    tool_args = creator.create(
        **task_to_kwargs(task=task, task_tools=task_tools, agent=agent, summarizer_agent=summarizer_agent),
        response_model=tool_type,  # type: ignore
    )
    logger.info(f"{tool.name} tool_args: {tool_args}")
    tool_args_message = assistant_message(content=f"Generated for {tool.name}: {tool_args}")
    task_tool.tool_args_messages += [tool_args_message]
    await add_task_tool_args_messages(
        task_tool_id=task_tool.id, tool_args_messages=[tool_args_message], db_name=db_name
    )
    res = await _run_selected_tool(selected_tool=selected_tool, tool_args=tool_args, is_async=is_async)
    logger.success(f"{tool.name} result: {res}")
    if isinstance(res, Task):
        if res.name in [agent.name, task.name]:
            return res
        task_messages = await get_task_messages(task=res, db_name=db_name, used=True)
        res = get_last_message(messages=task_messages, role="assistant")
        if res is not None:
            res = res["content"]
    else:
        logger.info(f"adding assistant message: {res}")
        res_message = assistant_message(content=str(res))
        task_tool.tool_result_messages += [res_message]
        await add_task_tool_result_messages(
            task_tool_id=task_tool.id, tool_result_messages=[res_message], db_name=db_name
        )
    # await save_agent(agent=agent, db_name=db_name)
    # await save_task(task=task, db_name=db_name)
    return task


async def run_selected_tool(
    db_name: str,
    task: Task,
    agent: Agent,
    task_tool: TaskTool,
    tool_is_agent: bool = False,
    creator: Instructor | AsyncInstructor | None = None,
    available_tools: list[str] | set[str] | None = None,
    summarizer_agent: Agent | None = SUMMARIZER_AGENT,
    tools_module: str = DEFAULT_TOOLS_MODULE,
) -> Task:
    # task = await load_task(task=task, db_name=db_name)
    # agent = await load_agent(agent=agent, db_name=db_name, current_task_id=task.id)
    creator = creator or create_creator(model_name=agent.model_name)
    if task_tool.tool.name.lower() == "ask_user":
        task = await run_ask_user(
            db_name=db_name,
            task=task,
            agent=agent,
            task_tool=task_tool,
            creator=creator,
            available_tools=available_tools,
            summarizer_agent=summarizer_agent,
        )
    elif task_tool.tool.name.lower() == "call_ai":
        task = await run_call_ai(
            db_name=db_name,
            task=task,
            agent=agent,
            task_tool=task_tool,
            creator=creator,
            summarizer_agent=summarizer_agent,
        )
    else:
        task = await run_tool(
            db_name=db_name,
            task=task,
            agent=agent,
            task_tool=task_tool,
            tool_is_agent=tool_is_agent,
            available_tools=available_tools,
            creator=creator,
            summarizer_agent=summarizer_agent,
            tools_module=tools_module,
        )
    return task


async def run_tools(
    db_name: str,
    task: Task,
    agent: Agent,
    creator: Instructor | AsyncInstructor | None = None,
    available_tools: list[str] | set[str] | None = None,
    agents: list[str] | set[str] | None = None,
    continue_dialog: bool = CONTINUE_DIALOG,
    summarizer_agent: Agent | None = SUMMARIZER_AGENT,
    tools_module: str = DEFAULT_TOOLS_MODULE,
) -> Task:
    # task = await load_task(task=task, db_name=db_name)
    # agent = await load_agent(agent=agent, db_name=db_name, current_task_id=task.id)
    creator = creator or create_creator(model_name=agent.model_name)
    while True:
        next_tool = await get_next_task_tool(task=task, db_name=db_name)
        logger.info(f"next_tool: {next_tool}")
        if next_tool is None:
            break
        attempts = 0
        while attempts < TOOL_ATTEMPT_LIMIT:
            try:
                task = await run_selected_tool(
                    db_name=db_name,
                    task=task,
                    creator=creator,
                    agent=agent,
                    task_tool=next_tool,
                    tool_is_agent=next_tool.tool.name in agents if agents else False,
                    available_tools=available_tools,
                    summarizer_agent=summarizer_agent,
                    tools_module=tools_module,
                )
                await set_task_tool_used(task_tool_id=next_tool.id, db_name=db_name)
                break
            except Exception as e:
                logger.exception(f"Error running tool '{next_tool.tool.name}'")
                await handle_creator_error(task=task, db_name=db_name, error=e, tool_name=next_tool.tool.name)
                attempts += 1
            await save_task(task=task, db_name=db_name)
    used_tools = await get_task_tools(task=task, db_name=db_name, used=True)
    last_used_tool = used_tools[-1].tool if used_tools else DEFAULT_TOOL
    if continue_dialog:
        task = await _continue_dialog(
            db_name=db_name,
            task=task,
            agent=agent,
            seq=int(last_used_tool.name == "ask_user"),
            last_used_tool=last_used_tool,
            creator=creator,
            available_tools=available_tools,
            summarizer_agent=summarizer_agent,
        )

    created_agent_tool_signature = await get_agent_tool_signature(db_name=db_name, task=task)
    agent_tool_signature = agent.agent_tool_signature or created_agent_tool_signature
    if agent_tool_signature is not None:
        agent.agent_tool_signature = agent_tool_signature
        await save_agent(agent=agent, db_name=db_name, agent_tool_signature=agent_tool_signature)
    await save_task(task=task, db_name=db_name)
    return task


async def new_task(
    db_name: str,
    agent: UUID4 | str | Agent,
    task_query: str,
    task_name: str = "",
    task: Task | None = None,
    creator: Instructor | AsyncInstructor | None = None,
    available_tools: list[str] | set[str] | None = None,
    recommended_tools: dict[int, str] | list[str] | None = None,
    selected_tools: list[Tool] | None = None,
    continue_dialog: bool = CONTINUE_DIALOG,
    summarizer_agent: Agent | None = SUMMARIZER_AGENT,
    tools_module: str = DEFAULT_TOOLS_MODULE,
) -> Task:
    agent = await load_agent(agent=agent, db_name=db_name)
    task = task or Task(name=task_name, agent_id=agent.id, task_query=task_query)
    agent = await load_agent(agent=agent, db_name=db_name, current_task_id=task.id)
    tools = import_module(tools_module)
    agents = await get_agents_with_signatures(db_name=db_name)
    agent_names = []
    agents_as_tools = []
    for a in agents:
        if a.name != agent.name:
            agent_names.append(a.name)
            agents_as_tools.append(f"Tool:\n{a.agent_tool_signature.to_dict()}")  # type: ignore
    tool_names = []
    tools_list = []
    available_tools = set(available_tools or agent.available_tools or [])
    selected_tools = selected_tools or []
    for _, func in get_functions_from_module(module=tools):
        if not available_tools or func.__name__ in available_tools | PINNED_TOOLS | set(
            tool.name for tool in selected_tools
        ):
            tool_names.append(func.__name__)
            tools_list.append(f"Tool:\n{get_function_info(func)}")
    tools_info = "\n\n".join(tools_list + agents_as_tools)

    av_message = user_message(
        content=f"<available_tools_for_task>\n{tools_info}\n</available_tools_for_task>",
    )
    task.available_tools_message = av_message
    if recommended_tools is None:
        used_tools = await get_task_tools(task=task, db_name=db_name, used=True)
        recommended_tools = {tool.position: tool.tool.name for tool in used_tools}
    if recommended_tools:
        rec_message = user_message(
            content=f"<recommended_tools_for_task_in_order>\n{json.dumps(recommended_tools, indent=2)}\n</recommended_tools_for_task_in_order>",
        )
        task.recommended_tools_message = rec_message
    await save_task(task=task, db_name=db_name)
    await add_task_tools(
        task=task,
        tools=selected_tools or Tool(reasoning="", name="get_selected_tools", prompt=task_query, db_name=db_name),  # type: ignore
    )
    return await run_tools(
        db_name=db_name,
        task=task,
        agent=agent,
        creator=creator,
        available_tools=set(tool_names) | set(agent_names) | PINNED_TOOLS,
        agents=set(agent_names),
        continue_dialog=continue_dialog,
        summarizer_agent=summarizer_agent,
        tools_module=tools_module,
    )
