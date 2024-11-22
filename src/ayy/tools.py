from typing import Any, Literal

import numpy as np


def call_ai(inputs: Any) -> Any:
    "Not a pre-defined tool."
    return inputs


def ask_user(prompt: str) -> str:
    "Prompt the user for a response"
    return prompt


def get_random_number() -> int:
    "get a random number"
    return np.random.randint(0, 100)


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
