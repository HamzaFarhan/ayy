from loguru import logger
from tortoise import run_async

from ayy.dialog import Dialog, ModelName
from ayy.dialogs import SUMMARIZER_DIALOG
from ayy.leggo import new_task
from ayy.torm import init_db

MODEL_NAME = ModelName.GEMINI_FLASH


DB_NAME = "tasks_db"
APP_NAME = "tasks"


async def setup():
    await init_db(db_names=DB_NAME, app_names=APP_NAME)
    # await save_dialog(
    #     dialog=Dialog(model_name=MODEL_NAME, name="default_dialog"), db_name=DB_NAME, overwrite=False
    # )


if __name__ == "__main__":
    logger.info("Setting up")
    run_async(setup())
    logger.success("Setup done")
    logger.info("Running task")
    run_async(
        new_task(
            db_name=DB_NAME,
            dialog=Dialog(model_name=MODEL_NAME, name="exp", max_message_tokens=300),
            task_query="list the grounds in london",
            # task_query="list the grounds in manchester and the weather there on tue",
            # task_query="weather on tuesday?",
            summarizer_dialog=SUMMARIZER_DIALOG,
        )
    )
    logger.success("Task done")