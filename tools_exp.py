from pathlib import Path
from typing import Literal
from valkey import Valkey

from ayy.dialog import Dialog, ModelName, create_creator
from ayy.tools2 import add_new_tools, run_tools, new_task

MODEL_NAME = ModelName.GEMINI_FLASH


def get_weather(
    day: Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], location: str
) -> str:
    "get the weather at a day in a location"
    if day == "Monday" and location.lower() == "blackpool":
        return "It's raining"
    elif day == "Tuesday" and location.lower() == "london":
        return "It's sunny"
    else:
        return "It's overcast"


def list_available_grounds(location: str) -> list[str]:
    "list all available grounds in a city"
    if location.lower() == "blackpool":
        return ["The Hawthorns", "The Den", "The New Den"]
    elif location.lower() == "london":
        return ["The Olympic Stadium", "The Emirates Stadium", "The Wembley Stadium"]
    else:
        return ["The Stadium"]


def upload_video(video_path: str) -> str:
    "save video and return the dest path"
    return video_path


# Initialize Valkey client
valkey_client = Valkey()

add_new_tools(valkey_client, new_tools=[get_weather, list_available_grounds, upload_video])
creator = create_creator(model_name=MODEL_NAME)
dialog = new_task(
    valkey_client,
    dialog=Dialog(system=Path("selector_task.txt").read_text(), model_name=MODEL_NAME),
    task="get the weather in blackpool",
)

runner_dialog = run_tools(valkey_client, creator=creator, dialog=dialog)
