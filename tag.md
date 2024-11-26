## Suggested Memory Tags and Implementation

### **Additional Memory Tags**

Expanding your current tagging system can provide finer control over how messages are handled within the dialog. Here are some suggested tags:

1. **`core`**: 
   - **Description**: Messages that are crucial and should persist even after the dialog concludes.
   - **Use Case**: Important facts, user preferences, or critical instructions.

2. **`recall`**: 
   - **Description**: Messages relevant to the current task and should be remembered during the ongoing dialog.
   - **Use Case**: Temporary information required to complete a specific task.

3. **`temporary`**:
   - **Description**: Messages that are only needed for a short duration and can be discarded afterward.
   - **Use Case**: Intermediate steps or clarifications that are not essential in the long term.

4. **`context`**:
   - **Description**: Provides background information that aids in understanding the conversation.
   - **Use Case**: Previous interactions, situational details, or environmental context.

5. **`action`**:
   - **Description**: Represents actions taken or to be taken.
   - **Use Case**: Commands issued, tools invoked, or processes initiated.

6. **`feedback`**:
   - **Description**: User or system feedback intended to improve future interactions.
   - **Use Case**: User ratings, corrections, or suggestions.

7. **`error`**:
   - **Description**: Records of errors or issues encountered during the dialog.
   - **Use Case**: Exception messages, failed operations, or system alerts.

### **Implementation Strategy**

To effectively manage these tags, it's advisable to **extend each message to be an object** rather than a simple dictionary. This approach allows for greater flexibility and scalability in handling additional metadata such as tags.

#### **1. Updating the `MessageType`**

Modify the `MessageType` to include a `tags` field. This will allow each message to carry its own set of tags.

```python
# src/ayy/dialog.py

from typing import List

class MessageType(BaseModel):
    role: str
    content: str
    tags: List[str] = Field(default_factory=list)
```

#### **2. Modifying Message Creation Functions**

Update the message creation functions to accept an optional `tags` parameter.

```python
# src/ayy/dialog.py

def chat_message(role: str, content: Content, template: Content = "", tags: List[str] = None) -> MessageType:
    if template:
        if not isinstance(content, dict):
            raise TypeError("When using template, content must be a dict.")
        try:
            message_content = template.format(**content)
        except KeyError as e:
            raise KeyError(f"Template {template} requires key {e} which was not found in content.")
    else:
        message_content = content
    return MessageType(role=role, content=message_content, tags=tags or [])
```

```python
# src/ayy/dialog.py

def system_message(content: Content, template: Content = "", tags: List[str] = None) -> MessageType:
    return chat_message(role="system", content=content, template=template, tags=tags)

def user_message(content: Content, template: Content = "", tags: List[str] = None) -> MessageType:
    return chat_message(role="user", content=content, template=template, tags=tags)

def assistant_message(content: Content, template: Content = "", tags: List[str] = None) -> MessageType:
    return chat_message(role="assistant", content=content, template=template, tags=tags)
```

#### **3. Updating the `Dialog` Model**

Ensure that the `Dialog` model accommodates the updated `MessageType`.

```python
# src/ayy/dialog.py

class Dialog(BaseModel):
    system: Content = ""
    messages: List[MessageType] = Field(default_factory=list)
    model_name: str = MODEL_NAME
    creation_config: dict = dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    # Removed memory_tags as tags are now per message

    def to_sql_dialog(self) -> SQL_Dialog:
        return SQL_Dialog(
            system=self.system, 
            model_name=self.model_name, 
            messages=[SQL_Message(**msg.dict()) for msg in self.messages]
        )
```

#### **4. Managing Tags When Adding Messages**

When adding messages to the dialog, specify appropriate tags based on the context.

```python
# src/ayy/leggo.py

def add_message_to_dialog(dialog: Dialog, role: str, content: str, tags: List[str] = None) -> Dialog:
    if tags is None:
        tags = []
    if role == "system":
        dialog.system = content
    else:
        message = None
        if role == "user":
            message = user_message(content=content, tags=tags)
        elif role == "assistant":
            message = assistant_message(content=content, tags=tags)
        elif role == "system":
            message = system_message(content=content, tags=tags)
        if message:
            dialog.messages.append(message)
    return dialog
```

#### **5. Filtering Messages Based on Tags**

Implement utility functions to retrieve messages based on their tags for various purposes like memory management or logging.

```python
# src/ayy/dialog.py

def get_messages_by_tag(dialog: Dialog, tag: str) -> List[MessageType]:
    return [msg for msg in dialog.messages if tag in msg.tags]

def get_core_messages(dialog: Dialog) -> List[MessageType]:
    return get_messages_by_tag(dialog, "core")

def get_recall_messages(dialog: Dialog) -> List[MessageType]:
    return get_messages_by_tag(dialog, "recall")
```

### **Advantages of Per-Message Tagging**

- **Granular Control**: Allows each message to have specific tags, enabling precise filtering and processing.
- **Scalability**: Easily extendable to include more tags without affecting the overall structure.
- **Flexibility**: Facilitates diverse use-cases by associating multiple tags with a single message.

### **Considerations**

- **Backward Compatibility**: Ensure that existing messages without tags are handled gracefully, possibly by assigning default tags or ignoring the tag-related functionalities for them.
- **Performance**: While adding tags increases the message size, it provides negligible overhead unless dealing with extremely large dialogs.

### **Example Usage**

Here’s how you can add messages with tags to the dialog:

```python
# exp.py

from ayy.dialog import Dialog
from ayy.leggo import add_message_to_dialog

dialog = Dialog(
    system="Initial system prompt.",
    model_name=ModelName.GEMINI_FLASH
)

# Adding a core message
dialog = add_message_to_dialog(dialog, role="assistant", content="System initialized.", tags=["core"])

# Adding a recall message
dialog = add_message_to_dialog(dialog, role="user", content="What's the weather in London?", tags=["recall"])

# Adding a temporary message
dialog = add_message_to_dialog(dialog, role="assistant", content="Fetching weather data...", tags=["temporary"])
```

### **Conclusion**

Implementing per-message tagging offers a robust and flexible way to manage memory within your dialog system. It allows for precise control over which messages are retained, recalled, or discarded based on their associated tags. This approach aligns well with your current architecture and enhances the system's capability to handle complex interactions effectively.
