from loguru import logger
from pydantic import UUID4
from tortoise import run_async

from ayy.dialog import DEFAULT_DIALOG, TAGGER_DIALOG, Dialog, ModelName
from ayy.leggo import new_task
from ayy.torm import init_db, save_dialog

MODEL_NAME = ModelName.GEMINI_FLASH


DB_NAME = "tasks_db"
APP_NAME = "tasks"


async def _new_task(
    dialog: UUID4 | str | Dialog, task: str, task_name: str = "", tagger_dialog: UUID4 | str | Dialog | None = None
) -> Dialog:
    return await new_task(
        db_name=DB_NAME, dialog=dialog, task=task, task_name=task_name, tagger_dialog=tagger_dialog
    )


async def setup():
    await init_db(db_names=DB_NAME, app_names=APP_NAME)
    await save_dialog(dialog=DEFAULT_DIALOG, db_name=DB_NAME)
    await save_dialog(dialog=TAGGER_DIALOG, db_name=DB_NAME)


if __name__ == "__main__":
    logger.info("Setting up")
    run_async(setup())
    logger.success("Setup done")
    logger.info("Running task")
    run_async(
        _new_task(
            dialog="default_dialog",
            task="list the grounds in manchester",
            task_name="list_grounds",
            tagger_dialog="tagger_dialog",
        )
    )
    logger.success("Task done")
