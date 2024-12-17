from fastapi import FastAPI, HTTPException
from pydantic import UUID4, BaseModel

from ayy.dialog import Dialog, ModelName, Tool
from ayy.leggo import new_task as create_new_task
from ayy.torm import add_task_tools, load_dialog, load_task

app = FastAPI()


class NewTaskRequest(BaseModel):
    task_query: str
    dialog_name: str = ""
    model_name: ModelName = ModelName.GEMINI_FLASH
    max_message_tokens: int = 300


class UserMessageRequest(BaseModel):
    task_id: UUID4
    message: str


DB_NAME = "tasks_db"


@app.post("/task/new")
async def new_task(request: NewTaskRequest):
    try:
        task, dialog = await create_new_task(
            db_name=DB_NAME,
            dialog=Dialog(
                model_name=request.model_name,
                name=request.dialog_name,
                max_message_tokens=request.max_message_tokens,
            ),
            task_query=request.task_query,
        )
        return {"task_id": task.id, "dialog_id": dialog.id, "messages": task.messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/task/{task_id}/message")
async def add_user_message(task_id: UUID4, request: UserMessageRequest):
    try:
        task = await load_task(task=task_id, db_name=DB_NAME)
        dialog = await load_dialog(dialog=task.dialog_id, db_name=DB_NAME)

        # Add call_ai tool with user's message as prompt
        await add_task_tools(
            task=task,
            tools=[Tool(reasoning="User message continuation", name="call_ai", prompt=request.message)],
            db_name=DB_NAME,
            run_next=True,  # This ensures it runs immediately
        )

        return {"task_id": task.id, "dialog_id": dialog.id, "messages": task.messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
