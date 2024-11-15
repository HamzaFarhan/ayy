import inspect
import typing
from functools import partial
from typing import Callable

from pydantic import BaseModel, create_model

from ayy.utils import deindent


def get_param_names(func: Callable):
    func = func.func if isinstance(func, partial) else func
    return inspect.signature(func).parameters.keys()


def get_required_param_names(func: Callable) -> list[str]:
    if isinstance(func, partial):
        params = inspect.signature(func.func).parameters
        return [
            name
            for name, param in params.items()
            if param.default == inspect.Parameter.empty and name not in func.keywords.keys()
        ]
    params = inspect.signature(func).parameters
    return [name for name, param in params.items() if param.default == inspect.Parameter.empty]


def function_schema(func: Callable) -> dict:
    kw = {
        n: (o.annotation, ... if o.default == inspect.Parameter.empty else o.default)
        for n, o in inspect.signature(func).parameters.items()
    }
    s = create_model(f"Input for `{func.__name__}`", **kw).schema()  # type: ignore
    return dict(name=func.__name__, description=func.__doc__, parameters=s)


def function_to_model(func: Callable) -> type[BaseModel]:
    kw = {
        n: (
            str if o.annotation == inspect.Parameter.empty else o.annotation,
            ... if o.default == inspect.Parameter.empty else o.default,
        )
        for n, o in inspect.signature(func).parameters.items()
    }
    return create_model(func.__name__, __doc__=func.__doc__, **kw)  # type:ignore


def get_function_return_type(func: Callable) -> type:
    func = func.func if isinstance(func, partial) else func
    sig = typing.get_type_hints(func)
    return sig.get("return", None)


def get_function_name(func: Callable) -> str:
    func = func.func if isinstance(func, partial) else func
    return func.__name__


def get_function_source(func: Callable) -> str:
    func = func.func if isinstance(func, partial) else func
    return inspect.getsource(func)


def get_function_info(func: Callable) -> dict[str, str]:
    func = func.func if isinstance(func, partial) else func
    name = func.__name__
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func)
    info = {"name": name}
    if signature:
        info["signature"] = f"{name}{signature}"
    if docstring:
        info["docstring"] = deindent(docstring)
    return info
