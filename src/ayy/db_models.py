from tortoise import fields, models

from ayy.agent import MAX_MESSAGE_TOKENS, MAX_TOKENS, MODEL_NAME, TEMPERATURE

DEFAULT_APP_NAME = "tasks"


class Agent(models.Model):
    id = fields.UUIDField(pk=True)
    name = fields.CharField(max_length=255, default="")
    system = fields.TextField(default="")
    messages = fields.JSONField(default=list)
    model_name = fields.CharField(max_length=255, default=MODEL_NAME.value)
    max_message_tokens = fields.IntField(default=MAX_MESSAGE_TOKENS)
    creation_config = fields.JSONField(default=dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS))
    available_tools = fields.JSONField(default=list)
    include_tool_guidelines = fields.BooleanField(default=True)
    summarized_tasks = fields.JSONField(default=list)
    agent_tool_signature = fields.JSONField(default=dict)

    class Meta:  # type: ignore
        app = DEFAULT_APP_NAME
        table = "agent"


class Task(models.Model):
    id = fields.UUIDField(pk=True)
    name = fields.CharField(max_length=255, default="")
    agent = fields.ForeignKeyField(f"{DEFAULT_APP_NAME}.Agent", related_name="task")
    task_query = fields.TextField()
    available_tools_message = fields.JSONField(default=dict)
    recommended_tools_message = fields.JSONField(default=dict)
    selected_tools_message = fields.JSONField(default=dict)
    summary = fields.JSONField(default=dict)
    summarized_task_tools = fields.JSONField(default=list)

    class Meta:  # type: ignore
        app = DEFAULT_APP_NAME
        table = "task"


class TaskTool(models.Model):
    id = fields.IntField(pk=True)
    task = fields.ForeignKeyField(f"{DEFAULT_APP_NAME}.Task", related_name="task_tool")
    position = fields.IntField()
    tool = fields.JSONField(default=dict)
    tool_args_messages = fields.JSONField(default=list)
    tool_result_messages = fields.JSONField(default=list)
    used = fields.BooleanField(default=False)
    created_at = fields.DatetimeField(auto_now_add=True)
    used_at = fields.DatetimeField(null=True)

    class Meta:  # type: ignore
        app = DEFAULT_APP_NAME
        table = "task_tool"
        ordering = ["task_id", "position"]


class SemanticMemoryDB(models.Model):
    id = fields.UUIDField(pk=True)
    name = fields.CharField(max_length=255)
    content = fields.TextField()
    category = fields.CharField(max_length=255)
    confidence = fields.FloatField(default=1.0)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    last_agent = fields.ForeignKeyField(f"{DEFAULT_APP_NAME}.Agent", related_name="semantic_memory", null=True)
    last_task = fields.ForeignKeyField(f"{DEFAULT_APP_NAME}.Task", related_name="semantic_memory", null=True)

    class Meta:  # type: ignore
        app = DEFAULT_APP_NAME
        table = "semantic_memory"
