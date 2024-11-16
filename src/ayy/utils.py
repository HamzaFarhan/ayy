import inspect
import re
import textwrap
from itertools import chain
from pathlib import Path
from typing import Iterable


def flatten(o: Iterable):
    for item in o:
        if isinstance(item, str):
            yield item
            continue
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def to_camel(s: str, sep: str = "_") -> str:
    if sep not in s:
        return s
    return "".join(s.title().split(sep))


def to_snake(s: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def noop(x=None, *args, **kwargs):  # noqa
    return x


def resolve_data_path(data_path: list[str | Path] | str | Path, file_extension: str | None = None) -> chain:
    if not isinstance(data_path, list):
        data_path = [data_path]
    paths = []
    for dp in flatten(data_path):
        if isinstance(dp, (str, Path)):
            dp = Path(dp)
            if not dp.exists():
                raise Exception(f"Path {dp} does not exist.")
            if dp.is_dir():
                if file_extension:
                    paths.append(dp.glob(f"*.{file_extension}"))
                else:
                    paths.append(dp.iterdir())
            else:
                if file_extension is None or dp.suffix == f".{file_extension}":
                    paths.append([dp])
    return chain(*paths)


def flatten_list(my_list: list) -> list:
    new_list = []
    for x in my_list:
        if isinstance(x, list):
            new_list += flatten_list(x)
        else:
            new_list.append(x)
    return new_list


def deindent(text: str) -> str:
    return textwrap.dedent(inspect.cleandoc(text))


def remove_digits(text: str) -> str:
    return re.sub(r"\d+", "", text)
