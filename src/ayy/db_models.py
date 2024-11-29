from tortoise import fields, models

from ayy.dialog import MAX_TOKENS, MODEL_NAME, TEMPERATURE

DEFAULT_APP_NAME = "tasks"


class Tool(models.Model):
    id = fields.UUIDField(pk=True)
    reasoning = fields.TextField()
    name = fields.CharField(max_length=255)
    prompt = fields.TextField()

    class Meta:  # type: ignore
        app = DEFAULT_APP_NAME
        table = "tools"


class Dialog(models.Model):
    id = fields.UUIDField(pk=True)
    system = fields.TextField(default="")
    messages = fields.JSONField(default=list)
    model_name = fields.CharField(max_length=255, default=MODEL_NAME.value)
    creation_config = fields.JSONField(default=dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS))

    class Meta:  # type: ignore
        app = DEFAULT_APP_NAME
        table = "dialog"


class ToolUsage(models.Model):
    id = fields.UUIDField(pk=True)
    tool = fields.ForeignKeyField(f"{DEFAULT_APP_NAME}.Tool", related_name="usage_entries")
    dialog = fields.ForeignKeyField(f"{DEFAULT_APP_NAME}.Dialog", related_name="usage_entries")
    position = fields.IntField()
    used = fields.BooleanField(default=False)
    timestamp = fields.DatetimeField(auto_now_add=True)

    class Meta:  # type: ignore
        app = DEFAULT_APP_NAME
        table = "tool_queue"
        ordering = ["dialog_id", "position"]
