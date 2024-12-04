from loguru import logger
from pydantic import UUID4
from tortoise import run_async

from ayy.dialog import Dialog, ModelName
from ayy.dialogs import MEMORY_TAGGER_DIALOG
from ayy.leggo import new_task
from ayy.torm import init_db, save_dialog

MODEL_NAME = ModelName.GEMINI_FLASH


DB_NAME = "tasks_db"
APP_NAME = "tasks"


async def _new_task(
    dialog: UUID4 | str | Dialog,
    task: str,
    memory_tagger_dialog: UUID4 | str | Dialog | None = None,
    available_tools: list[str] | set[str] | None = None,
) -> Dialog:
    return await new_task(
        db_name=DB_NAME,
        dialog=dialog,
        task=task,
        memory_tagger_dialog=memory_tagger_dialog,
        available_tools=available_tools,
    )


async def setup():
    await init_db(db_names=DB_NAME, app_names=APP_NAME)
    await save_dialog(
        dialog=Dialog(model_name=MODEL_NAME, name="default_dialog"), db_name=DB_NAME, overwrite=False
    )
    await save_dialog(dialog=MEMORY_TAGGER_DIALOG, db_name=DB_NAME, overwrite=False)


if __name__ == "__main__":
    logger.info("Setting up")
    run_async(setup())
    logger.success("Setup done")
    logger.info("Running task")
    run_async(
        _new_task(
            dialog="list_grounds",
            # task="list the grounds in london",
            task="list the grounds in manchester and the weather there on tue",
            memory_tagger_dialog=MEMORY_TAGGER_DIALOG,
            available_tools=["list_available_grounds_and_format_response_and_get_weather_and_format_response"],
        )
    )
    logger.success("Task done")
