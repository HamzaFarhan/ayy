import json
from functools import partial
from pathlib import Path
from typing import Any, Literal

import instructor
from google.generativeai import GenerativeModel
from loguru import logger
from pydantic import Field, create_model

from ayy.dialog import Dialog, ModelName, assistant_message, messages_to_kwargs, user_message
from ayy.func_utils import get_function_info

MODEL = ModelName.GEMINI_PRO

days = Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def get_weather(day: days, location: str) -> str:
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


def call_ai(inputs: Any) -> Any:
    """
    Not a pre-defined tool.
    Flow:
        1. Get some inputs from either the user or the result of a previous tool call.
        2. Send the inputs to the AI/LLM/Assistant
        3. Return the response
    When to use:
        - Whenever you want. In between or at the start/end.
        - To get the AI to generate something. This could be the final response or something in between tool calls.
        - To extract information before the next tool call.
    """
    return inputs


def ask_user(prompt: str) -> str:
    """
    Prompt the user for a response
    When to use:
        - To ask the user something.
        - When the right tool is obvious but you need some extra data based on the function signature, then select the ask_user tool before selecting the actual tool and in your prompt explicitly state the extra information you need. So the tool should still be selected, you're just pairing it with a prior ask_user call.
        - A tool may have some default values. But if you think they should be provided by the user for the current task, ask for them.
    Don't use the 'ask_user' tool to ask the user to fix obvious typos, do that yourself. That's a safe assumption. People make typos all the time.
    """
    return prompt


tools = [call_ai, ask_user, get_weather, list_available_grounds, upload_video]
tools_dict = {func.__name__: func for func in tools}
SelectedTool = create_model(
    "SelectedTool",
    chain_of_thought=(str, ...),
    name=(Literal[*tools_dict.keys()], ...),
    prompt=(
        str,
        Field(
            ...,
            description="An LLM will receive the messages so far and the tools calls and results up until now. This prompt will then be used to ask the LLM to generate an input for the selected tool based on the tool's signature.",
        ),
    ),
)
tools_info = "\n\n".join([f"Tool {i}:\n{get_function_info(func)}" for i, func in enumerate(tools, start=1)])

tools_message = """
You have a list tools at your disposal. Each tool is a function with a signature and optional docstring.
Based on the user query, return a list of tools to use for the task. The tools will be used in that sequence.
You can assume that a tool would have access to the result of a previous tool call.
The list can also be empty. For each tool selection, return the tool name and a prompt for the LLM to generate an input for the selected tool based on the tool's signature. The LLM will receive the messages so far and the tools calls and results up until that point.


<tools>

{tools_info}

</tools>
"""

plan_dialog = Dialog(
    system=tools_message.format(tools_info=tools_info),
    messages=[
        user_message(
            "I want to play a football match, what are the available grounds? I won't play if it's raining"
        )
    ],
    model_name=MODEL,
)
plan_path = Path("plan.json")
if not plan_path.exists():
    plan_creator = partial(
        instructor.from_gemini(client=GenerativeModel(model_name=MODEL), mode=instructor.Mode.GEMINI_JSON).create,
        response_model=list[SelectedTool],  # type:ignore
    )
    plan = plan_creator(
        **messages_to_kwargs(messages=plan_dialog.messages, system=plan_dialog.system, model_name=MODEL),
        generation_config=dict(temperature=0.1),
    )
    logger.success(f"Plan: {plan}")
    plan_path.write_text(json.dumps([selection.model_dump() for selection in plan], indent=2))  # type:ignore
else:
    plan = json.loads(plan_path.read_text())
plan_dialog.messages.append(assistant_message(f"I will use these tools in order: {plan}"))
logger.info(plan_dialog.messages)
