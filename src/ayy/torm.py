from datetime import datetime
from importlib import import_module
from typing import Any

from loguru import logger
from pydantic import UUID4, BaseModel
from tortoise import Tortoise, connections
from tortoise.contrib.pydantic import pydantic_model_creator, pydantic_queryset_creator
from tortoise.expressions import F
from tortoise.transactions import in_transaction

from ayy.agent import Agent, AgentToolSignature, Messages
from ayy.db_models import DEFAULT_APP_NAME
from ayy.db_models import Agent as DBAgent
from ayy.db_models import Task as DBTask
from ayy.db_models import TaskTool as DBTaskTool
from ayy.func_utils import get_functions_from_module
from ayy.task import Task, TaskTool, Tool, task_to_messages, task_tool_to_messages

DEFAULT_DB_NAME = "tasks_db"
DEFAULT_TOOLS_MODULE = "ayy.tools"


def db_task_tool_to_task_tool(db_task_tool: BaseModel | dict) -> TaskTool:
    if isinstance(db_task_tool, BaseModel):
        db_task_tool = db_task_tool.model_dump()
    db_task_tool["task_id"] = db_task_tool["task"]["id"]
    return TaskTool(**db_task_tool)


async def init_db(db_names: list[str] | str = DEFAULT_DB_NAME, app_names: list[str] | str = DEFAULT_APP_NAME):
    db_names = [db_names] if isinstance(db_names, str) else db_names
    app_names = [app_names] if isinstance(app_names, str) else app_names
    assert len(db_names) == len(app_names), "Number of db_names and app_names must be the same"
    await Tortoise.init(
        config={
            "connections": {db_name: f"sqlite://{db_name}.sqlite3" for db_name in db_names},
            "apps": {
                app_name: {"models": ["ayy.db_models"], "default_connection": db_name}
                for app_name, db_name in zip(app_names, db_names)
            },
        }
    )
    await Tortoise.generate_schemas()


async def get_next_task_tool(
    task: UUID4 | Task,
    db_name: str = DEFAULT_DB_NAME,
    used: bool = False,
    reverse: bool = False,
    position: int | None = None,
) -> TaskTool | None:
    task_id = task.id if isinstance(task, Task) else task
    task_tools = DBTaskTool.filter(task_id=task_id).using_db(connections.get(db_name))
    if position is not None:
        task_tools = await task_tools.filter(position=position).first()
    else:
        task_tools = (
            await task_tools.filter(used=used).order_by("position" if not reverse else "-position").first()
        )
    if task_tools is None:
        return None
    tool_model = pydantic_model_creator(DBTaskTool)
    tool_model = await tool_model.from_tortoise_orm(await task_tools)
    return db_task_tool_to_task_tool(db_task_tool=tool_model)


async def get_next_available_task_tool_id_and_position(
    task: UUID4 | Task, db_name: str = DEFAULT_DB_NAME
) -> tuple[int, int]:
    task_id = task.id if isinstance(task, Task) else task
    task_tools = DBTaskTool.filter(task_id=task_id).using_db(connections.get(db_name))
    latest_tool = await task_tools.order_by("-id").first()
    return 1 if latest_tool is None else latest_tool.id + 1, 1 if latest_tool is None else latest_tool.position + 1


async def get_agents_with_signatures(db_name: str = DEFAULT_DB_NAME) -> list[Agent]:
    agents = await pydantic_queryset_creator(DBAgent).from_queryset(
        DBAgent.filter(agent_tool_signature__not={}).using_db(connections.get(db_name))
    )
    return [Agent(**agent) for agent in agents.model_dump()]


async def get_task_tools(
    task: UUID4 | Task, db_name: str = DEFAULT_DB_NAME, used: bool | None = False, max_position: int | None = None
) -> list[TaskTool]:
    filter_kwargs: dict[str, Any] = {}
    if used is not None:
        filter_kwargs["used"] = used
    if max_position is not None:
        filter_kwargs["position__lte"] = max_position
    task_tools = await pydantic_queryset_creator(DBTaskTool).from_queryset(
        DBTaskTool.filter(task_id=task.id if isinstance(task, Task) else task, **filter_kwargs)
        .order_by("position")
        .using_db(connections.get(db_name))
    )
    return [db_task_tool_to_task_tool(db_task_tool=tool) for tool in task_tools.model_dump()]


async def add_task_tools(
    task: UUID4 | Task,
    tools: list[Tool | TaskTool] | Tool | TaskTool,
    db_name: str = DEFAULT_DB_NAME,
    position: int | None = None,
    used: bool = False,
    run_next: bool = False,
    replace_all: bool = False,
) -> None:
    tools = [tools] if not isinstance(tools, list) else tools
    conn = connections.get(db_name)
    task_id = task.id if isinstance(task, Task) else task
    async with in_transaction():
        conn = connections.get(db_name)
        task_tools = DBTaskTool.filter(task_id=task_id).using_db(conn)
        if replace_all:
            await task_tools.delete()
            start_position = 1
        else:
            if run_next:
                first_unused = await task_tools.filter(used=False).order_by("position").first()
                start_position = 1 if first_unused is None else first_unused.position
                await task_tools.filter(position__gte=start_position).update(position=F("position") + len(tools))
            else:
                query = task_tools.order_by("-position")
                latest_tool = await query.first()
                if latest_tool is None:
                    latest_position = 0
                else:
                    latest_position = latest_tool.position
                logger.info(f"latest_position: {latest_position}")
                if position is not None:
                    await task_tools.filter(position__gte=position).update(position=F("position") + len(tools))
                    start_position = position
                else:
                    start_position = latest_position + 1
        for i, tool in enumerate(tools, start=start_position):
            if isinstance(tool, Tool):
                await DBTaskTool.create(
                    using_db=conn,
                    task_id=task_id,
                    position=i,
                    used=used,
                    tool=tool.model_dump(),
                )
            else:
                existing_tool = await DBTaskTool.filter(id=tool.id).using_db(conn).first()
                if existing_tool:
                    await existing_tool.update_from_dict(tool.model_dump())
                    await existing_tool.save()
                else:
                    await DBTaskTool.create(using_db=conn, **tool.model_dump())


async def toggle_task_tool_usage(task_tool_id: int, db_name: str = DEFAULT_DB_NAME) -> None:
    tool = await DBTaskTool.get(id=task_tool_id, using_db=connections.get(db_name))
    tool.used = not tool.used
    await tool.save()


async def set_task_tool_used(task_tool_id: int, db_name: str = DEFAULT_DB_NAME) -> None:
    tool = await DBTaskTool.get(id=task_tool_id, using_db=connections.get(db_name))
    tool.used = True
    tool.used_at = datetime.now()
    await tool.save()


async def add_task_tool_args_messages(
    task_tool_id: int, tool_args_messages: Messages, db_name: str = DEFAULT_DB_NAME
) -> None:
    tool = await DBTaskTool.get(id=task_tool_id, using_db=connections.get(db_name))
    tool.tool_args_messages += tool_args_messages
    await tool.save()


async def add_task_tool_result_messages(
    task_tool_id: int, tool_result_messages: Messages, db_name: str = DEFAULT_DB_NAME
) -> None:
    tool = await DBTaskTool.get(id=task_tool_id, using_db=connections.get(db_name))
    tool.tool_result_messages += tool_result_messages
    # tool.used = True
    # tool.used_at = datetime.now()
    await tool.save()


async def save_task(task: Task, db_name: str = DEFAULT_DB_NAME, overwrite: bool = True) -> None:
    conn = connections.get(db_name)
    existing_task = await DBTask.filter(id=task.id).using_db(conn).first()
    if existing_task is None and task.name != "":
        existing_task = await DBTask.filter(name=task.name).using_db(conn).first()
    if existing_task is None:
        await DBTask.create(using_db=conn, **task.model_dump())
    elif overwrite:
        existing_task = await existing_task.update_from_dict(
            {k: v for k, v in task.model_dump().items() if k not in ["id"]}
        )
        await existing_task.save()


async def load_task(task: UUID4 | str | Task, db_name: str = DEFAULT_DB_NAME) -> Task:
    if isinstance(task, Task):
        return task
    conn = connections.get(db_name)
    kwargs = {"name": task} if isinstance(task, str) else {"id": task}
    task_obj, _ = await DBTask.get_or_create(defaults=None, using_db=conn, **kwargs)
    task_model = pydantic_model_creator(DBTask)
    task_model = await task_model.from_tortoise_orm(task_obj)
    task_model_dump = task_model.model_dump()
    task_model_dump["agent_id"] = task_model_dump["agent"]["id"]
    return Task(**task_model_dump)


async def get_task_tools_messages(
    task: UUID4 | Task, db_name: str = DEFAULT_DB_NAME, used: bool | None = None
) -> Messages:
    task_tools = await get_task_tools(task=task, db_name=db_name, used=used)
    return [msg for task_tool in task_tools for msg in task_tool_to_messages(task_tool)]


async def get_task_messages(
    task: UUID4 | Task, db_name: str = DEFAULT_DB_NAME, used: bool | None = None, summarized: bool = False
) -> Messages:
    task = await load_task(task=task, db_name=db_name)
    task_tools = await get_task_tools(task=task, db_name=db_name, used=used)
    return task_to_messages(task=task, task_tools=task_tools, summarized=summarized)


async def save_agent(
    agent: Agent,
    db_name: str = DEFAULT_DB_NAME,
    agent_tool_signature: AgentToolSignature | None = None,
    tools_module: str = DEFAULT_TOOLS_MODULE,
    overwrite: bool = True,
) -> None:
    conn = connections.get(db_name)
    agent_dict = agent.model_dump()
    if agent_tool_signature is not None:
        if agent_tool_signature.name not in [f[0] for f in get_functions_from_module(import_module(tools_module))]:
            agent_dict["system"] = agent_tool_signature.system
            agent_dict["name"] = agent_tool_signature.name
            agent_dict["agent_tool_signature"] = agent_tool_signature.model_dump()
    existing_agent = await DBAgent.filter(name=agent.name).using_db(conn).first() if agent.name != "" else None
    if existing_agent is None:
        existing_agent = await DBAgent.filter(id=agent.id).using_db(conn).first()
    if existing_agent is None:
        await DBAgent.create(using_db=conn, **agent_dict)
    elif overwrite:
        existing_agent = await existing_agent.update_from_dict(
            {k: v for k, v in agent_dict.items() if k not in ["id", "agent_id"]}
        )
        await existing_agent.save()


async def load_agent(
    agent: UUID4 | str | Agent,
    db_name: str = DEFAULT_DB_NAME,
    current_task_id: UUID4 | None = None,
    load_task_messages: bool = True,
) -> Agent:
    conn = connections.get(db_name)
    kwargs = {}
    defaults = None
    if isinstance(agent, Agent):
        kwargs["name"] = agent.name
        defaults = agent.model_dump()
    elif isinstance(agent, str):
        kwargs["name"] = agent
    else:
        kwargs["id"] = agent
    agent_obj, _ = await DBAgent.get_or_create(defaults=defaults, using_db=conn, **kwargs)
    agent_model = pydantic_model_creator(DBAgent)
    agent_model = await agent_model.from_tortoise_orm(agent_obj)
    agent = Agent(**agent_model.model_dump())
    if not load_task_messages:
        return agent
    filter_ids = [
        task_id for task_id in agent.summarized_tasks if current_task_id is None or task_id != current_task_id
    ]
    tasks = await DBTask.filter(agent_id=agent.id, id__not_in=filter_ids).using_db(conn).all()

    for task in tasks:
        task_messages = await get_task_messages(task=task.id, db_name=db_name, used=True, summarized=True)
        agent.messages += task_messages

    return agent
