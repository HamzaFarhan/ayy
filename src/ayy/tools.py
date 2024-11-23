from typing import Any, Literal

from pydantic import BaseModel, Field

from ayy.dialog import DEFAULT_PROMPT


class Tool(BaseModel):
    chain_of_thought: str
    name: str
    prompt: str = Field(
        ...,
        description="An LLM will receive the messages so far and the tools calls and results up until now. This prompt will then be used to ask the LLM to generate arguments for the selected tool based on the tool's signature. If the tool doesn't have any parameters, then it doesn't need a prompt.",
    )

    def __str__(self) -> str:
        return f"Chain Of Thought: {self.chain_of_thought}\nName: {self.name}\nPrompt: {self.prompt}"


DEFAULT_TOOL = Tool(chain_of_thought="", name="call_ai", prompt=DEFAULT_PROMPT)


def call_ai(inputs: Any) -> Any:
    "Not a pre-defined tool."
    return inputs


def ask_user(prompt: str) -> str:
    "Prompt the user for a response"
    return prompt


def get_weather(
    day: Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], location: str
) -> str:
    "get the weather at a day in a location"
    if day == "Monday" and location.lower() == "blackpool":
        return "It's raining"
    elif day == "Tuesday" and location.lower() == "london":
        return "It's sunny"
    elif day == "Wednesday" and location.lower() == "manchester":
        return "It's cloudy"
    else:
        return "It's overcast"


def list_available_grounds(location: str) -> list[str]:
    "list all available grounds in a location"
    if location.lower() == "blackpool":
        return ["The Hawthorns", "The Den", "The New Den"]
    elif location.lower() == "london":
        return ["Wembley Stadium", "Emirates Stadium", "Tottenham Hotspur Stadium"]
    elif location.lower() == "manchester":
        return ["The Old Trafford", "The Etihad Stadium", "The Maine Road"]
    else:
        return ["Palm Football"]


def download_video(url: str) -> str:
    "download video from url and return the local path"
    return f"videos/{url.split('/')[-1]}"
