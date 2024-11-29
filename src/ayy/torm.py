from collections import deque
from typing import Deque

from tortoise import Tortoise
from tortoise.exceptions import DoesNotExist
from tortoise.transactions import in_transaction

from ayy.db_models import CurrentTool, ToolQueue
from ayy.db_models import Tool as DBTool
from ayy.dialog import DEFAULT_PROMPT
from ayy.tools import DEFAULT_TOOL, Tool


async def init(name: str):
    await Tortoise.init(
        db_url=f"sqlite://{name}.sqlite3",
        modules={"models": ["ayy.db_models"]},
    )
    await Tortoise.generate_schemas()


async def get_tool_queue() -> Deque[Tool]:
    tool_queue = deque()
    async for tq in ToolQueue.all().order_by("position"):
        tool_queue.append(Tool(**tq.tool))
    return tool_queue


async def get_current_tool() -> str:
    try:
        current_tool = await CurrentTool.first()
        return current_tool.tool.name if current_tool else DEFAULT_TOOL.name
    except DoesNotExist:
        return DEFAULT_TOOL.name


async def update_tool_queue(tool_queue: Deque[Tool]):
    async with in_transaction():
        await ToolQueue.all().delete()
        for position, tool in enumerate(tool_queue, start=1):
            tool_obj, _ = await DBTool.get_or_create(
                name=tool.name, defaults={"chain_of_thought": tool.chain_of_thought, "prompt": tool.prompt}
            )
            await ToolQueue.create(tool=tool_obj, position=position)


async def pop_next_tool() -> Tool:
    tool_queue = await get_tool_queue()
    if not tool_queue:
        # Handle empty queue by returning DEFAULT_TOOL
        return DEFAULT_TOOL
    tool = tool_queue.popleft()
    await update_tool_queue(tool_queue)
    return tool


async def update_current_tool(tool_name: str):
    tool_obj, _ = await DBTool.get_or_create(
        name=tool_name,
        defaults={"chain_of_thought": "", "prompt": DEFAULT_PROMPT},
    )
    current_tool, created = await CurrentTool.get_or_create(id=1, defaults={"tool": tool_obj})
    if not created:
        current_tool.tool = tool_obj
        await current_tool.save()
