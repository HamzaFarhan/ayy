from tortoise import fields, models

from ayy.dialog import MAX_TOKENS, MODEL_NAME, TEMPERATURE

DEFAULT_APP_NAME = "tasks"


class Dialog(models.Model):
    id = fields.UUIDField(pk=True)
    system = fields.TextField(default="")
    messages = fields.JSONField(default=list)
    model_name = fields.CharField(max_length=255, default=MODEL_NAME.value)
    creation_config = fields.JSONField(default=dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS))
    name = fields.CharField(max_length=255, default="")
    dialog_tool_signature = fields.JSONField(default=dict)
    available_tools = fields.JSONField(default=list)

    class Meta:  # type: ignore
        app = DEFAULT_APP_NAME
        table = "dialog"


# class DialogTool(models.Model):
#     id = fields.IntField(pk=True)
#     dialog = fields.ForeignKeyField(f"{DEFAULT_APP_NAME}.Dialog", related_name="dialog_tool")
#     position = fields.IntField()
#     reasoning = fields.TextField()
#     name = fields.CharField(max_length=255)
#     prompt = fields.TextField()
#     used = fields.BooleanField(default=False)
#     timestamp = fields.DatetimeField(auto_now_add=True)

#     class Meta:  # type: ignore
#         app = DEFAULT_APP_NAME
#         table = "dialog_tool"
#         ordering = ["dialog_id", "position"]


class Task(models.Model):
    id = fields.UUIDField(pk=True)
    name = fields.CharField(max_length=255, default="")
    dialog = fields.ForeignKeyField(f"{DEFAULT_APP_NAME}.Dialog", related_name="task")
    messages = fields.JSONField(default=list)

    class Meta:  # type: ignore
        app = DEFAULT_APP_NAME
        table = "task"


class TaskTool(models.Model):
    id = fields.IntField(pk=True)
    task = fields.ForeignKeyField(f"{DEFAULT_APP_NAME}.Task", related_name="task_tool")
    position = fields.IntField()
    reasoning = fields.TextField()
    name = fields.CharField(max_length=255)
    prompt = fields.TextField()
    used = fields.BooleanField(default=False)
    timestamp = fields.DatetimeField(auto_now_add=True)

    class Meta:  # type: ignore
        app = DEFAULT_APP_NAME
        table = "task_tool"
        ordering = ["task_id", "position"]
