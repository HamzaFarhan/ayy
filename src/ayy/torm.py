from importlib import import_module

from loguru import logger
from pydantic import UUID4, BaseModel
from tortoise import Tortoise, connections
from tortoise.contrib.pydantic import pydantic_model_creator, pydantic_queryset_creator
from tortoise.expressions import F
from tortoise.transactions import in_transaction

from ayy.db_models import DEFAULT_APP_NAME
from ayy.db_models import Dialog as DBDialog
from ayy.db_models import Task as DBTask
from ayy.db_models import TaskTool as DBTaskTool
from ayy.dialog import Dialog, DialogToolSignature, Task, TaskTool, Tool
from ayy.func_utils import get_functions_from_module

DEFAULT_DB_NAME = "tasks_db"
TOOL_FIELDS = ["reasoning", "name", "prompt"]
DEFAULT_TOOLS_MODULE = "ayy.tools"


def db_task_tool_to_task_tool(db_task_tool: BaseModel | dict, tool_fields: list[str] = TOOL_FIELDS) -> TaskTool:
    if isinstance(db_task_tool, BaseModel):
        db_task_tool = db_task_tool.model_dump()
    return TaskTool(
        id=db_task_tool["id"],
        position=db_task_tool["position"],
        task_id=db_task_tool["task"]["id"],
        tool=Tool(**{k: v for k, v in db_task_tool.items() if k in tool_fields}),
        used=db_task_tool["used"],
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


async def get_dialogs_with_signatures(db_name: str = DEFAULT_DB_NAME) -> list[Dialog]:
    dialogs = await pydantic_queryset_creator(DBDialog).from_queryset(
        DBDialog.filter(dialog_tool_signature__not={}).using_db(connections.get(db_name))
    )
    return [Dialog(**dialog) for dialog in dialogs.model_dump()]


async def get_task_tools(task: UUID4 | Task, db_name: str = DEFAULT_DB_NAME, used: bool = False) -> list[TaskTool]:
    task_id = task.id if isinstance(task, Task) else task
    task_tools = await pydantic_queryset_creator(DBTaskTool).from_queryset(
        DBTaskTool.filter(task_id=task_id, used=used).order_by("position").using_db(connections.get(db_name))
    )
    return [db_task_tool_to_task_tool(db_task_tool=tool) for tool in task_tools.model_dump()]


async def add_task_tools(
    task: UUID4 | Task,
    tools: list[Tool] | Tool,
    db_name: str = DEFAULT_DB_NAME,
    position: int | None = None,
    used: bool = False,
    run_next: bool = False,
    replace_all: bool = False,
    tool_fields: list[str] = TOOL_FIELDS,
) -> None:
    tools = [tools] if isinstance(tools, Tool) else tools
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
            logger.info(f"tool: {tool}")
            await DBTaskTool.create(
                using_db=conn, task_id=task_id, position=i, used=used, **tool.model_dump(include=set(tool_fields))
            )


async def toggle_task_tool_usage(task_tool_id: int, db_name: str = DEFAULT_DB_NAME) -> None:
    tool = await DBTaskTool.get(id=task_tool_id, using_db=connections.get(db_name))
    tool.used = not tool.used
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
    return Task(**task_model.model_dump())


async def save_dialog(
    dialog: Dialog,
    db_name: str = DEFAULT_DB_NAME,
    dialog_tool_signature: DialogToolSignature | None = None,
    tools_module: str = DEFAULT_TOOLS_MODULE,
    overwrite: bool = True,
) -> None:
    conn = connections.get(db_name)
    dialog_dict = dialog.model_dump()
    if dialog_tool_signature is not None:
        if dialog_tool_signature.name not in [
            f[0] for f in get_functions_from_module(import_module(tools_module))
        ]:
            dialog_dict["system"] = dialog_tool_signature.system
            dialog_dict["name"] = dialog_tool_signature.name
            dialog_dict["dialog_tool_signature"] = dialog_tool_signature.model_dump()
    existing_dialog = await DBDialog.filter(name=dialog.name).using_db(conn).first() if dialog.name != "" else None
    if existing_dialog is None:
        existing_dialog = await DBDialog.filter(id=dialog.id).using_db(conn).first()
    if existing_dialog is None:
        await DBDialog.create(using_db=conn, **dialog_dict)
    elif overwrite:
        existing_dialog = await existing_dialog.update_from_dict(
            {k: v for k, v in dialog_dict.items() if k not in ["id", "dialog_id"]}
        )
        await existing_dialog.save()


async def load_dialog(dialog: UUID4 | str | Dialog, db_name: str = DEFAULT_DB_NAME) -> Dialog:
    if isinstance(dialog, Dialog):
        return dialog
    conn = connections.get(db_name)
    kwargs = {"name": dialog} if isinstance(dialog, str) else {"id": dialog}
    dialog_obj, _ = await DBDialog.get_or_create(defaults=None, using_db=conn, **kwargs)
    dialog_model = pydantic_model_creator(DBDialog)
    dialog_model = await dialog_model.from_tortoise_orm(dialog_obj)
    return Dialog(**dialog_model.model_dump())
