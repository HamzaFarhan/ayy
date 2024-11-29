from tortoise import fields, models


class Tool(models.Model):
    id = fields.UUIDField(pk=True)
    name = fields.CharField(max_length=255, unique=True)
    chain_of_thought = fields.TextField()
    prompt = fields.TextField()

    class Meta:  # type: ignore
        table = "tools"


class ToolQueue(models.Model):
    id = fields.UUIDField(pk=True)
    tool = fields.ForeignKeyField("db_models.Tool", related_name="queue_entries")
    position = fields.IntField()

    class Meta:  # type: ignore
        table = "tool_queue"
        ordering = ["position"]


class CurrentTool(models.Model):
    id = fields.UUIDField(pk=True)
    tool = fields.ForeignKeyField("db_models.Tool", related_name="current_tool")

    class Meta:  # type: ignore
        table = "current_tool"
