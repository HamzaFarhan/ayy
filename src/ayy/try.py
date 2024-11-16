import json
from pathlib import Path
from typing import Any

import pandas as pd

from ayy.dialog import Dialog, user_message


def jsonl_to_json(src: str | Path) -> Any:
    json_obj = pd.read_json(path_or_buf=Path(src).with_suffix(".jsonl"), lines=True)
    return json.loads(str(json_obj.to_json(indent=2, orient="records")))


def add_anns_to_dialog(project_dir: str | Path, app_id: str) -> Dialog:
    project_dir = Path(project_dir)
    log_seq_ids = []
    logs = {}
    for log in jsonl_to_json(src=project_dir / app_id / "log.jsonl"):
        if log["type"] == "end_entry":
            logs[log["sequence_id"]] = log
            log_seq_ids.append(log["sequence_id"])

    initial_state = logs[min(log_seq_ids)]["state"]
    final_state = logs[max(log_seq_ids)]["state"]
    initial_num_messages = len(initial_state["messages"])

    num_added = 0
    for ann in jsonl_to_json(src=project_dir / "annotations.jsonl"):
        if ann["app_id"] == app_id:
            for obs in ann["observations"]:
                thumbs = obs.pop("thumbs_up_thumbs_down", None)
                obs.update(
                    {
                        "thumbs_up": bool(thumbs) if thumbs is not None else False,
                        "thumbs_down": not bool(thumbs) if thumbs is not None else False,
                    }
                )
            final_state["messages"].insert(
                ann["step_sequence_id"] + initial_num_messages + num_added,
                user_message(content=f"<observations>\n{json.dumps(ann['observations'])}\n</observations>"),
            )
            num_added += 1

    return Dialog(**final_state)


dialog = add_anns_to_dialog(project_dir="~/.burr/EXP", app_id="exp2")
print(dialog)
