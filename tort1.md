Yes, you can implement the tool queue system using Tortoise ORM with SQLite. Based on the search results, Tortoise ORM is an async ORM that works well with SQLite. Here's how you could refactor the system:

First, let's create the models:

```python:src/ayy/models.py
from tortoise import fields, models
from typing import Optional
import dill
from ayy.tools import Tool, DEFAULT_TOOL

class ToolQueue(models.Model):
    id = fields.IntField(pk=True)
    position = fields.IntField()
    tool_data = fields.BinaryField()  # For storing serialized Tool objects
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "tool_queue"
        ordering = ["position"]

class CurrentTool(models.Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "current_tool"
```

Then modify the VK functions to use the database:

```python:src/ayy/vk.py
from typing import Deque
from collections import deque
import dill
from tortoise import Tortoise
from ayy.models import ToolQueue, CurrentTool
from ayy.tools import Tool, DEFAULT_TOOL

async def init_db():
    await Tortoise.init(
        db_url='sqlite://db.sqlite3',
        modules={'models': ['ayy.models']}
    )
    await Tortoise.generate_schemas()

async def get_tool_queue() -> Deque:
    queue_items = await ToolQueue.all().order_by('position')
    queue = deque()
    for item in queue_items:
        tool = dill.loads(item.tool_data)
        queue.append(tool)
    return queue

async def get_current_tool() -> str:
    current = await CurrentTool.first()
    return current.name if current else DEFAULT_TOOL.name

async def update_tool_queue(tool_queue: Deque):
    # Clear existing queue
    await ToolQueue.all().delete()
    
    # Insert new queue
    for position, tool in enumerate(tool_queue):
        if isinstance(tool, Tool):
            tool = Tool(**tool.model_dump())
        await ToolQueue.create(
            position=position,
            tool_data=dill.dumps(tool)
        )

async def pop_next_tool() -> Tool:
    first_item = await ToolQueue.first()
    if not first_item:
        return DEFAULT_TOOL
        
    tool = dill.loads(first_item.tool_data)
    await first_item.delete()
    
    # Reorder remaining items
    await ToolQueue.filter(position__gt=first_item.position).update(
        position=fields.F('position') - 1
    )
    
    return tool

async def update_current_tool(tool_name: str):
    current = await CurrentTool.first()
    if current:
        current.name = tool_name
        await current.save()
    else:
        await CurrentTool.create(name=tool_name)
```

The main changes are:

1. Created Tortoise ORM models for the tool queue and current tool
2. Converted functions to be async to work with the async ORM
3. Added proper database initialization
4. Used binary fields to store serialized Tool objects
5. Added position tracking for queue order
6. Added timestamps for auditing

You'll need to modify the calling code to handle the async nature of these functions. For example:

```python:src/ayy/leggo.py
async def run_tools(
    creator: Instructor | AsyncInstructor,
    dialog: Dialog,
    continue_dialog: bool = True,
    available_tools: list[str] | set[str] | None = None,
    tag_messages: bool = TAG_MESSAGES,
) -> Dialog:
    tool_queue = await get_tool_queue()
    current_tool_name = await get_current_tool()
    
    if not tool_queue:
        tool_queue = deque([DEFAULT_TOOL])
        await update_tool_queue(tool_queue=tool_queue)

    while tool_queue:
        tools_str = "\n\n".join([str(tool) for tool in tool_queue])
        logger.info(f"\nTOOL QUEUE:\n\n{tools_str}\n")
        current_tool = await pop_next_tool()
        # ... rest of the function ...
```

This implementation provides several benefits:

1. Persistence - Tools survive program restarts
2. Atomic operations - Database transactions ensure queue integrity
3. Easier debugging - Can inspect queue state directly in database
4. Better scalability - Can handle larger queues efficiently
5. Audit trail - Timestamps track when tools were added/updated

Remember to initialize the database before using these functions:

```python
await init_db()
```

The code assumes you have Tortoise ORM installed (`pip install tortoise-orm[sqlite]`). Let me know if you need any clarification or have questions about specific parts of the implementation!
