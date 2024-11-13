import json
from functools import partial
from pathlib import Path
from typing import Any, Literal

import instructor
from google.generativeai import GenerativeModel
from loguru import logger
from pydantic import Field, create_model

from ayy.dialog import Dialog, ModelName, assistant_message, messages_to_kwargs, user_message
from ayy.func_utils import function_to_model, get_function_info

MODEL = ModelName.GEMINI_FLASH

days = Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def get_weather(day: days, location: str) -> str:
    "get the weather at a day in a location"
    if day == "Monday" and location.lower() == "blackpool":
        return "It's raining"
    elif day == "Tuesday" and location.lower() == "london":
        return "It's sunny"
    else:
        return "It's overcast"


def list_available_grounds(location: str) -> str:
    "list all available grounds in a city"
    if location.lower() == "blackpool":
        return json.dumps(["The Hawthorns", "The Den", "The New Den"])
    elif location.lower() == "london":
        return json.dumps(["The Olympic Stadium", "The Emirates Stadium", "The Wembley Stadium"])
    else:
        return json.dumps(["The Stadium"])


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
Remember the actual user query/task throughout your tool selection process. Especially when creating the prompt for the LLM.
More often than not, the last tool would be 'call_ai' to generate the final response.
Pay close attention to the information you do have and the information you do not have. If you don't have the information and you think the user should know, ask the user for it. Don't make assumptions.


<tools>

{tools_info}

</tools>
"""

creator = partial(
    instructor.from_gemini(
        client=GenerativeModel(model_name=MODEL), mode=instructor.Mode.GEMINI_JSON
    ).chat.messages.create,
    generation_config=dict(temperature=0.1),
)

messages = [user_message("I want to buy a new mac")]
plan_dialog = Dialog(system=tools_message.format(tools_info=tools_info), messages=messages, model_name=MODEL)
plan_path = Path("plan.json")
if not plan_path.exists():
    plan = creator(
        **messages_to_kwargs(messages=plan_dialog.messages, system=plan_dialog.system, model_name=MODEL),
        response_model=list[SelectedTool],
        generation_config=dict(temperature=0.1),
    )
    logger.success(f"Plan: {plan}")
    plan_path.write_text(json.dumps([selection.model_dump() for selection in plan], indent=2))  # type:ignore
else:
    plan = [SelectedTool(**tool) for tool in json.loads(plan_path.read_text())]


for tool in plan:  # type:ignore
    if tool.name == "ask_user":
        res = input(f"{tool.prompt}\n> ")
        messages += [assistant_message(content=tool.prompt), user_message(content=res)]
    elif tool.name == "call_ai":
        messages.append(user_message(content=tool.prompt))
        logger.info(f"\n\nCalling AI with messages: {messages}\n\n")
        res = creator(**messages_to_kwargs(messages=messages, model_name=MODEL), response_model=str)
        logger.success(f"call_ai result: {res}")
        messages.append(assistant_message(content=res))
    else:
        messages.append(user_message(content=tool.prompt))
        logger.info(f"\n\nCalling Tool with messages: {messages}\n\n")
        tool_args = creator(
            **messages_to_kwargs(messages=messages, model_name=MODEL),
            response_model=function_to_model(tools_dict[tool.name]),
        )
        logger.info(f"Tool args: {tool_args}")
        res = tools_dict[tool.name](**tool_args.model_dump())  # type:ignore
        logger.success(f"Tool result: {res}")
        messages.append(assistant_message(content=res))

if not any(x.name for x in plan if x.name not in ["ask_user", "call_ai"]):  # type:ignore
    seq = 0
    while True:
        if seq % 2 == 0:
            user_input = input("> ")
            if user_input.lower() == "q":
                break
            messages.append(user_message(content=user_input))
        else:
            res = creator(**messages_to_kwargs(messages=messages, model_name=MODEL), response_model=str)
            logger.success(f"call_ai result: {res}")
            messages.append(assistant_message(content=res))
        seq += 1


logger.success(f"Messages: {messages}")
