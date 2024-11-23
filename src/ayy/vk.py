from collections import deque
from typing import Deque

import dill
from valkey import Valkey

from ayy.tools import DEFAULT_TOOL, Tool


def get_tool_queue(valkey_client: Valkey) -> Deque:
    queue = valkey_client.get("tool_queue")
    return deque(dill.loads(queue)) if queue else deque()  # type: ignore


def get_current_tool(valkey_client: Valkey) -> str:
    return valkey_client.get("current_tool_name") or DEFAULT_TOOL.name  # type: ignore


def update_tool_queue(valkey_client: Valkey, tool_queue: Deque):
    tools = []
    for tool in tool_queue:
        if isinstance(tool, Tool):
            tools.append(Tool(**tool.model_dump()))
        else:
            tools.append(tool)
    valkey_client.set("tool_queue", dill.dumps(tools))


def pop_next_tool(valkey_client: Valkey) -> Tool:
    tool_queue = get_tool_queue(valkey_client)
    tool = tool_queue.popleft()
    update_tool_queue(valkey_client=valkey_client, tool_queue=tool_queue)
    return tool


def update_current_tool(valkey_client: Valkey, tool_name: str):
    valkey_client.set("current_tool_name", tool_name)
