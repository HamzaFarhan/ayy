from pathlib import Path

from valkey import Valkey

from ayy.dialog import Dialog, ModelName, create_creator
from ayy.leggo import new_task, run_tools

MODEL_NAME = ModelName.GEMINI_FLASH

valkey_client = Valkey()
dialog = Dialog(system=Path("src/ayy/selector_task.txt").read_text(), model_name=MODEL_NAME)
creator = create_creator(model_name=MODEL_NAME)

dialog = new_task(valkey_client, dialog=dialog, task="whats the weather in london?")
dialog = run_tools(valkey_client, creator=creator, dialog=dialog)
