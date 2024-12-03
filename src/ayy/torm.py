from loguru import logger
from pydantic import UUID4, BaseModel
from tortoise import Tortoise, connections
from tortoise.contrib.pydantic import pydantic_model_creator, pydantic_queryset_creator
from tortoise.expressions import F
from tortoise.transactions import in_transaction

from ayy.db_models import DEFAULT_APP_NAME
from ayy.db_models import Dialog as DBDialog
from ayy.db_models import DialogTool as DBDialogTool
from ayy.dialog import Dialog, DialogTool, Tool

DEFAULT_DB_NAME = "tasks_db"
TOOL_FIELDS = ["reasoning", "name", "prompt"]


def db_dialog_tool_to_dialog_tool(db_dialog_tool: BaseModel | dict, fields: list[str] = TOOL_FIELDS) -> DialogTool:
    if isinstance(db_dialog_tool, BaseModel):
        db_dialog_tool = db_dialog_tool.model_dump()
    return DialogTool(
        id=db_dialog_tool["id"],
        position=db_dialog_tool["position"],
        dialog_id=db_dialog_tool["dialog"]["id"],
        tool=Tool(**{k: v for k, v in db_dialog_tool.items() if k in fields}),
        used=db_dialog_tool["used"],
    )


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


async def get_next_dialog_tool(
    dialog: UUID4 | Dialog, db_name: str = DEFAULT_DB_NAME, position: int | None = None
) -> DialogTool | None:
    dialog_id = dialog.id if isinstance(dialog, Dialog) else dialog
    dialog_tools = DBDialogTool.all(using_db=connections.get(db_name)).filter(dialog_id=dialog_id)
    if position is not None:
        dialog_tools = await dialog_tools.filter(position=position).first()
    else:
        dialog_tools = await dialog_tools.filter(used=False).order_by("position").first()
    if dialog_tools is None:
        return None
    tool_model = pydantic_model_creator(DBDialogTool)
    tool_model = await tool_model.from_tortoise_orm(await dialog_tools)
    return db_dialog_tool_to_dialog_tool(db_dialog_tool=tool_model)


async def get_dialog_tools(
    dialog: UUID4 | Dialog, db_name: str = DEFAULT_DB_NAME, used: bool = False
) -> list[DialogTool]:
    dialog_id = dialog.id if isinstance(dialog, Dialog) else dialog
    dialog_tools = await pydantic_queryset_creator(DBDialogTool).from_queryset(
        DBDialogTool.all(using_db=connections.get(db_name))
        .filter(dialog_id=dialog_id, used=used)
        .order_by("position")
    )
    return [db_dialog_tool_to_dialog_tool(db_dialog_tool=tool) for tool in dialog_tools.model_dump()]


async def add_dialog_tools(
    dialog: UUID4 | Dialog,
    tools: list[Tool] | Tool,
    db_name: str = DEFAULT_DB_NAME,
    position: int | None = None,
    replace_all: bool = False,
) -> None:
    tools = [tools] if isinstance(tools, Tool) else tools
    conn = connections.get(db_name)
    dialog_id = dialog.id if isinstance(dialog, Dialog) else dialog
    async with in_transaction():
        conn = connections.get(db_name)
        dialog_tools = DBDialogTool.all(using_db=conn).filter(dialog_id=dialog_id)
        if replace_all:
            await dialog_tools.delete()
            start_position = 1
        else:
            query = dialog_tools.order_by("-position")
            latest_tool = await query.first()
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
            await DBDialogTool.create(
                using_db=conn, dialog_id=dialog_id, position=i, **tool.model_dump(include=set(TOOL_FIELDS))
            )


async def toggle_dialog_tool_usage(dialog_tool_id: int, db_name: str = DEFAULT_DB_NAME) -> None:
    tool = await DBDialogTool.get(id=dialog_tool_id, using_db=connections.get(db_name))
    tool.used = not tool.used
    await tool.save()


async def save_dialog(dialog: Dialog, db_name: str = DEFAULT_DB_NAME) -> None:
    conn = connections.get(db_name)
    await DBDialog.update_or_create(defaults=dialog.model_dump(), using_db=conn, id=dialog.id)


async def load_dialog(dialog: UUID4 | str | Dialog, db_name: str = DEFAULT_DB_NAME) -> Dialog:
    if isinstance(dialog, Dialog):
        return dialog
    conn = connections.get(db_name)
    dialog_model = pydantic_model_creator(DBDialog)
    kwargs = {"name": dialog} if isinstance(dialog, str) else {"id": dialog}
    dialog_obj, created = await DBDialog.get_or_create(defaults=None, using_db=conn, **kwargs)
    if created:
        logger.warning(f"Dialog with query params {kwargs} not found, created new one")
    dialog_model = await dialog_model.from_tortoise_orm(dialog_obj)
    return Dialog(**dialog_model.model_dump())
