import pickle
from functools import partial
from typing import Any, Callable, Dict, Optional, Union

from instructor import AsyncInstructor, Instructor
from loguru import logger
from valkey import Valkey

# Key prefixes and utilities
KEY_PREFIX = "ayy:"


def make_key(key: str) -> str:
    return f"{KEY_PREFIX}{key}"


def serialize(obj: Any) -> bytes:
    return pickle.dumps(obj)


def deserialize(data: Optional[bytes]) -> Any:
    return pickle.loads(data) if data else None


# Tool queue operations
def push_tool(store: Valkey, tool: Union[Tool, Callable]) -> None:
    queue = deserialize(store.get(make_key("tool_queue"))) or []
    queue.append(tool)
    store.set(make_key("tool_queue"), serialize(queue))


def pop_tool(store: Valkey) -> Union[Tool, Callable, None]:
    queue = deserialize(store.get(make_key("tool_queue"))) or []
    return queue.pop() if queue else None


def get_queue_length(store: Valkey) -> int:
    queue = deserialize(store.get(make_key("tool_queue"))) or []
    return len(queue)


# Tool dictionary operations
def add_tool(store: Valkey, name: str, tool_info: dict) -> None:
    tools = deserialize(store.get(make_key("tool_dict"))) or {}
    tools[name] = tool_info
    store.set(make_key("tool_dict"), serialize(tools))


def get_tool(store: Valkey, name: str) -> dict:
    tools = deserialize(store.get(make_key("tool_dict"))) or {}
    return tools.get(name)


def get_all_tools(store: Valkey) -> Dict[str, dict]:
    return deserialize(store.get(make_key("tool_dict"))) or {}


# Current tool tracking
def set_current_tool(store: Valkey, name: str) -> None:
    store.set(make_key("current_tool"), name.encode())


def get_current_tool(store: Valkey) -> str:
    return store.get(make_key("current_tool")).decode()


# Storage management
def clear_storage(store: Valkey) -> None:
    store.clear()


def setup_default_tools(store: Valkey) -> None:
    """Initialize storage with default tools"""
    clear_storage(store)
    default_tools = {call_ai, ask_user}
    for func in default_tools:
        tool_info = {"info": get_function_info(func), "func": func}
        add_tool(store, func.__name__, tool_info)
    set_current_tool(store, DEFAULT_TOOL.name)


def add_new_tools(store: Valkey, new_tools: set[Callable] | list[Callable]) -> None:
    """Add new tools to storage"""
    for func in new_tools:
        tool_info = {"info": get_function_info(func), "func": func}
        add_tool(store, func.__name__, tool_info)

    # Add get_selected_tools with updated tool names
    tool_names = list(get_all_tools(store).keys())
    add_tool(
        store,
        "get_selected_tools",
        {
            "info": get_function_info(get_selected_tools),
            "func": get_selected_tools,
            "type": list[create_model("SelectedTool", name=(Literal[*tool_names], ...), __base__=Tool)],
        },
    )


def run_tools(
    store: Valkey, creator: Union[Instructor, AsyncInstructor], dialog: Dialog, continue_dialog: bool = True
) -> Dialog:
    """Run tools using Valkey storage"""
    if get_queue_length(store) == 0:
        push_tool(store, DEFAULT_TOOL)

    while get_queue_length(store) > 0:
        current_tool = pop_tool(store)

        # Handle callable tools
        if not isinstance(current_tool, Tool) and callable(current_tool):
            current_tool_name = (
                current_tool.__name__ if not isinstance(current_tool, partial) else current_tool.func.__name__
            )
            set_current_tool(store, current_tool_name)
            res = current_tool()
            if res:
                if isinstance(res, Dialog):
                    dialog = res
                else:
                    dialog.messages.append(assistant_message(content=str(res)))
            continue

        # Handle Tool instances
        current_tool_name = current_tool.name
        set_current_tool(store, current_tool_name)
        dialog = run_selected_tool(creator=creator, dialog=dialog, tool=current_tool)

    if continue_dialog:
        seq = int(get_current_tool(store) == "ask_user")
        while True:
            if seq % 2 == 0 or get_current_tool(store) == "call_ai":
                user_input = input("('q' or 'exit' or 'quit' to quit) > ")
                if user_input.lower() in ["q", "exit", "quit"]:
                    break
                push_tool(store, partial(new_task, dialog=dialog, task=user_input))
                dialog = run_tools(store, creator, dialog, continue_dialog=False)
            else:
                set_current_tool(store, "call_ai")
                res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
                logger.success(f"ai response: {res}")
                dialog.messages.append(assistant_message(content=res))
            seq += 1

    logger.success(f"Messages: {dialog.messages[-2:]}")
    return dialog
