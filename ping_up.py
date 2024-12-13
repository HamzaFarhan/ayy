from loguru import logger
from tortoise import run_async

from ayy.db_models import DEFAULT_APP_NAME
from ayy.dialog import ModelName
from ayy.dialogs import SUMMARIZER_DIALOG
from ayy.leggo import new_task
from ayy.torm import init_db, save_dialog
from ping_dialogs import (
    ASSET_ADVISOR_DIALOG,
    COMPLIANCE_OFFICER_DIALOG,
    CUSTOMER_SUPPORT_DIALOG,
    HUMAN_ADVISOR_DIALOG,
    INVESTOR_ASSISTANT_DIALOG,
    RISK_ANALYST_DIALOG,
)

MODEL_NAME = ModelName.GEMINI_FLASH


DB_NAME = "ping_db"
APP_NAME = DEFAULT_APP_NAME


async def setup():
    await init_db(db_names=DB_NAME, app_names=APP_NAME)
    await save_dialog(dialog=CUSTOMER_SUPPORT_DIALOG, db_name=DB_NAME, overwrite=False)
    await save_dialog(dialog=INVESTOR_ASSISTANT_DIALOG, db_name=DB_NAME, overwrite=False)
    await save_dialog(dialog=ASSET_ADVISOR_DIALOG, db_name=DB_NAME, overwrite=False)
    await save_dialog(dialog=RISK_ANALYST_DIALOG, db_name=DB_NAME, overwrite=False)
    await save_dialog(dialog=COMPLIANCE_OFFICER_DIALOG, db_name=DB_NAME, overwrite=False)
    await save_dialog(dialog=HUMAN_ADVISOR_DIALOG, db_name=DB_NAME, overwrite=False)
    await save_dialog(dialog=SUMMARIZER_DIALOG, db_name=DB_NAME, overwrite=False)


if __name__ == "__main__":
    logger.info("Setting up")
    run_async(setup())
    logger.success("Setup done")
    logger.info("Running task")
    run_async(
        new_task(
            db_name=DB_NAME,
            dialog="investor_assistant",
            task_query="What are some safe mutual funds I can invest in?",
        )
    )
    logger.success("Task done")
