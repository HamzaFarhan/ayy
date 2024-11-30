from loguru import logger
from pydantic import UUID4
from tortoise import run_async

from ayy.dialog import DEFAULT_DIALOG, TAGGER_DIALOG, Dialog, ModelName
from ayy.leggo import new_task, run_tools
from ayy.torm import init_db, save_dialog

MODEL_NAME = ModelName.GEMINI_FLASH


DB_NAME = "tasks_db"
APP_NAME = "tasks"


async def _new_task(dialog: UUID4 | Dialog | str, task: str):
    dialog = await new_task(db_name=DB_NAME, dialog=dialog, task=task)
    dialog = await run_tools(db_name=DB_NAME, dialog=dialog)


async def setup():
    await init_db(db_names=DB_NAME, app_names=APP_NAME)
    await save_dialog(dialog=DEFAULT_DIALOG, dialog_name="default_dialog", db_name=DB_NAME)
    await save_dialog(dialog=TAGGER_DIALOG, dialog_name="tagger_dialog", db_name=DB_NAME)


if __name__ == "__main__":
    logger.info("Setting up")
    run_async(setup())
    logger.success("Setup done")
    logger.info("Running task")
    run_async(_new_task(dialog="default_dialog", task="What is the weather in manchester?"))
    logger.success("Task done")
