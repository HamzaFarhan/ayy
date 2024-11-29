from pathlib import Path

from tortoise import run_async

from ayy.dialog import Dialog, ModelName, create_creator
from ayy.leggo import new_task, run_tools
from ayy.torm import DEFAULT_DB_NAME, init_db

MODEL_NAME = ModelName.GEMINI_FLASH


async def main():
    await init_db()
    creator = create_creator(model_name=MODEL_NAME)
    dialog = Dialog(system=Path("src/ayy/selector_task.txt").read_text(), model_name=MODEL_NAME)
    dialog = await new_task(db_name=DEFAULT_DB_NAME, dialog=dialog, task="whats the weather in london?")
    dialog = await run_tools(db_name=DEFAULT_DB_NAME, creator=creator, dialog=dialog)


if __name__ == "__main__":
    run_async(main())
