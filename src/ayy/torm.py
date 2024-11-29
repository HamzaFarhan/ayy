from collections import deque
from typing import Deque

from loguru import logger
from pydantic import UUID4
from tortoise import Tortoise, connections
from tortoise.contrib.pydantic import pydantic_model_creator, pydantic_queryset_creator
from tortoise.expressions import F
from tortoise.transactions import in_transaction

from ayy.db_models import DEFAULT_APP_NAME, ToolUsage
from ayy.db_models import Tool as DBTool
from ayy.dialog import DEFAULT_TOOL, Dialog, Tool

DEFAULT_DB_NAME = "tasks_db"


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


async def get_next_tool(dialog_id: UUID4, db_name: str = DEFAULT_DB_NAME, position: int | None = None) -> Tool:
    dialog_tools = ToolUsage.filter(dialog_id=dialog_id, using_db=connections.get(db_name))
    if position is not None:
        tool_usage = await dialog_tools.filter(position=position).first()
    else:
        tool_usage = await dialog_tools.filter(used=False).order_by("position").first()
    if tool_usage is None:
        return DEFAULT_TOOL
    tool_model = await pydantic_model_creator(DBTool).from_tortoise_orm(tool_usage.tool)
    return Tool(**tool_model.model_dump())


async def get_unused_tools(dialog_id: UUID4, db_name: str = DEFAULT_DB_NAME) -> list[Tool]:
    dialog_tools = await pydantic_queryset_creator(ToolUsage).from_queryset(
        ToolUsage.filter(dialog_id=dialog_id, used=False, using_db=connections.get(db_name)).order_by("position")
    )
    return [Tool(**tool["tool"].model_dump()) for tool in dialog_tools.model_dump()]


async def add_tool(
    dialog: UUID4 | Dialog, tools: list[Tool] | Tool, db_name: str = DEFAULT_DB_NAME, position: int | None = None
) -> None:
    conn = connections.get(db_name)
    tools = [tools] if isinstance(tools, Tool) else tools
    dialog_id = dialog.id if isinstance(dialog, Dialog) else dialog
    async with in_transaction():
        latest_tool = await ToolUsage.filter(dialog_id=dialog_id, using_db=conn).order_by("-position").first()
        latest_position = latest_tool.position if latest_tool is not None else 0
        start_position = latest_position + 1

        if position is not None:
            await ToolUsage.filter(dialog_id=dialog_id, position__gte=position, using_db=conn).update(
                position=F("position") + len(tools)
            )
            start_position = position

        for i, tool in enumerate(tools, start=min(start_position, 1)):
            tool_obj, _ = await DBTool.get_or_create(defaults=tool.model_dump(), id=tool.id, using_db=conn)
            await ToolUsage.create(tool=tool_obj, dialog_id=dialog_id, position=i, using_db=conn)


async def get_tool_queue(db_name: str = DEFAULT_DB_NAME) -> Deque[Tool]:
    db_tool_queue = await pydantic_queryset_creator(ToolQueue).from_queryset(
        ToolQueue.all(using_db=connections.get(db_name)).order_by("position")
    )
    logger.info(f"ToolQueue: {db_tool_queue.model_dump_json(indent=4)}")
    return deque(Tool(**entry["tool"]) for entry in db_tool_queue.model_dump())


async def get_current_tool(db_name: str = DEFAULT_DB_NAME) -> Tool:
    current_tool = await DBTool.all(using_db=connections.get(db_name)).order_by("-timestamp").first()
    if current_tool is None:
        return DEFAULT_TOOL
    current_tool = await pydantic_model_creator(DBTool).from_tortoise_orm(current_tool)
    logger.info(f"CurrentTool: {current_tool.model_dump_json(indent=4)}")
    return Tool(**current_tool.model_dump())  # type: ignore


async def update_tool_queue(tool_queue: Deque[Tool] | list[Tool] | Tool, db_name: str = DEFAULT_DB_NAME):
    tool_queue = [tool_queue] if isinstance(tool_queue, Tool) else tool_queue
    async with in_transaction():
        await ToolQueue.all(using_db=connections.get(db_name)).delete()
        for position, tool in enumerate(tool_queue, start=1):
            tool_obj, _ = await DBTool.get_or_create(defaults=tool.model_dump(), id=tool.id)
            await ToolQueue.create(using_db=connections.get(db_name), tool=tool_obj, position=position)


async def add_tools_to_queue(tools: list[Tool] | Tool, db_name: str = DEFAULT_DB_NAME, top: bool = True):
    tools = [tools] if isinstance(tools, Tool) else tools
    tool_queue = await get_tool_queue(db_name=db_name)
    if top:
        tool_queue.extendleft(tools[::-1])
    else:
        tool_queue.extend(tools)
    await update_tool_queue(tool_queue=tool_queue, db_name=db_name)


async def pop_next_tool(db_name: str = DEFAULT_DB_NAME, top: bool = True) -> Tool:
    tool_queue = await get_tool_queue(db_name=db_name)
    if not tool_queue:
        return DEFAULT_TOOL
    tool = tool_queue.popleft() if top else tool_queue.pop()
    await update_tool_queue(tool_queue=tool_queue, db_name=db_name)
    return tool


async def update_current_tool(tool_id: str, db_name: str = DEFAULT_DB_NAME):
    tool_obj, _ = await DBTool.get_or_create(
        defaults=DEFAULT_TOOL.model_dump(), using_db=connections.get(db_name), id=tool_id
    )
    current_tool, created = await CurrentTool.get_or_create(
        defaults={"tool": tool_obj}, using_db=connections.get(db_name), id=tool_obj.id
    )
    if not created:
        current_tool.tool = tool_obj
        await current_tool.save(using_db=connections.get(db_name))
