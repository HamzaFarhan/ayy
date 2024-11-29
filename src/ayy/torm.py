from loguru import logger
from pydantic import UUID4, BaseModel
from tortoise import Tortoise, connections
from tortoise.contrib.pydantic import pydantic_model_creator, pydantic_queryset_creator
from tortoise.expressions import F
from tortoise.transactions import in_transaction

from ayy.db_models import DEFAULT_APP_NAME, ToolUsage
from ayy.db_models import Dialog as DBDialog
from ayy.db_models import Tool as DBTool
from ayy.dialog import Dialog, Tool

DEFAULT_DB_NAME = "tasks_db"


class ToolWithPosition(BaseModel):
    position: int
    tool: Tool


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


async def get_next_tool(
    dialog: UUID4 | Dialog, db_name: str = DEFAULT_DB_NAME, position: int | None = None
) -> ToolWithPosition | None:
    dialog_id = dialog.id if isinstance(dialog, Dialog) else dialog
    dialog_tools = ToolUsage.all(using_db=connections.get(db_name)).filter(dialog_id=dialog_id)
    if position is not None:
        dialog_tools = await dialog_tools.filter(position=position).first()
    else:
        dialog_tools = await dialog_tools.filter(used=False).order_by("position").first()
    if dialog_tools is None:
        return None
    tool_model = pydantic_model_creator(DBTool)
    tool_model = await tool_model.from_tortoise_orm(await dialog_tools.tool)
    return ToolWithPosition(position=dialog_tools.position, tool=Tool(**tool_model.model_dump()))


async def get_tools(
    dialog: UUID4 | Dialog, db_name: str = DEFAULT_DB_NAME, used: bool = False
) -> list[ToolWithPosition]:
    dialog_id = dialog.id if isinstance(dialog, Dialog) else dialog
    dialog_tools = await pydantic_queryset_creator(ToolUsage).from_queryset(
        ToolUsage.all(using_db=connections.get(db_name))
        .filter(dialog_id=dialog_id, used=used)
        .order_by("position")
    )
    return [
        ToolWithPosition(position=tool["position"], tool=Tool(**tool["tool"]))
        for tool in dialog_tools.model_dump()
    ]


async def add_tools(
    dialog: UUID4 | Dialog,
    tools: list[Tool] | Tool,
    db_name: str = DEFAULT_DB_NAME,
    position: int | None = None,
) -> None:
    tools = [tools] if isinstance(tools, Tool) else tools
    conn = connections.get(db_name)
    # for tool in tools:
    #     tool_obj, _ = await DBTool.get_or_create(defaults=tool.model_dump(), using_db=conn, id=tool.id)
    dialog_id = dialog.id if isinstance(dialog, Dialog) else dialog
    async with in_transaction():
        conn = connections.get(db_name)
        dialog_tools = ToolUsage.all(using_db=conn).filter(dialog_id=dialog_id)
        query = dialog_tools.order_by("-position")
        logger.info(f"About to execute query: {query.sql()}")
        try:
            latest_tool = await query.first()
            logger.info(f"Query completed, latest_tool: {latest_tool}")
        except Exception as e:
            logger.error(f"Error in query execution: {str(e)}")
            logger.error(f"Current connection: {conn}")
            raise

        if latest_tool is None:
            latest_position = 0
        else:
            latest_position = latest_tool.position
        logger.info(f"latest_position: {latest_position}")
        if position is not None:
            await dialog_tools.filter(position__gte=position).update(position=F("position") + len(tools))
            start_position = position
        else:
            start_position = latest_position + 1
        for i, tool in enumerate(tools, start=start_position):
            logger.info(f"tool: {tool}")
            tool_obj, _ = await DBTool.get_or_create(defaults=tool.model_dump(), using_db=conn, id=tool.id)
            await ToolUsage.create(tool=tool_obj, dialog_id=dialog_id, position=i, using_db=conn)


async def set_tool_used(tool: Tool, dialog: UUID4 | Dialog, db_name: str = DEFAULT_DB_NAME) -> None:
    dialog_id = dialog.id if isinstance(dialog, Dialog) else dialog
    await (
        ToolUsage.all(using_db=connections.get(db_name))
        .filter(tool_id=tool.id, dialog_id=dialog_id)
        .update(used=True)
    )


async def save_dialog(dialog: Dialog, db_name: str = DEFAULT_DB_NAME) -> None:
    conn = connections.get(db_name)
    await DBDialog.update_or_create(defaults=dialog.model_dump(), using_db=conn, id=dialog.id)


async def load_dialog(dialog_id: UUID4, db_name: str = DEFAULT_DB_NAME) -> Dialog:
    conn = connections.get(db_name)
    dialog = pydantic_model_creator(DBDialog)
    dialog_obj, _ = await DBDialog.get_or_create(id=dialog_id, using_db=conn)
    if dialog_obj is None:
        raise ValueError(f"Dialog with id {dialog_id} not found")
    dialog_model = await dialog.from_tortoise_orm(dialog_obj)
    return Dialog(**dialog_model.model_dump())
