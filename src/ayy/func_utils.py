import inspect
import typing
from functools import partial
from inspect import Signature
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


def get_non_default_signature(func: Callable) -> Signature:
    if isinstance(func, partial):
        signature = inspect.signature(func.func)
        return signature.replace(
            parameters=[
                p
                for p in signature.parameters.values()
                if p.default == inspect.Parameter.empty and p.name not in func.keywords
            ]
        )
    else:
        signature = inspect.signature(func)
        return signature.replace(
            parameters=[p for p in signature.parameters.values() if p.default == inspect.Parameter.empty]
        )


def function_to_model(func: Callable | type, ignore_defaults: bool = False) -> type[BaseModel] | type:
    if isinstance(func, type):
        return func
    kw = {
        n: (
            str if o.annotation == inspect.Parameter.empty else o.annotation,
            ... if o.default == inspect.Parameter.empty else o.default,
        )
        for n, o in (
            get_non_default_signature(func) if ignore_defaults else inspect.signature(func)
        ).parameters.items()
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


def get_function_info(func: Callable, ignore_defaults: bool = False) -> dict[str, str]:
    func = func.func if isinstance(func, partial) else func
    name = func.__name__
    signature = get_non_default_signature(func) if ignore_defaults else inspect.signature(func)
    docstring = inspect.getdoc(func)
    info = {"name": name}
    if signature:
        info["signature"] = f"{name}{signature}"
    if docstring:
        info["docstring"] = deindent(docstring)
    return info


def get_weather(day: str, location: str) -> str:
    "get the weather at a day in a location"
    if day == "Monday" and location.lower() == "blackpool":
        return "It's raining"
    elif day == "Tuesday" and location.lower() == "london":
        return "It's sunny"
    else:
        return "It's overcast"


gw = partial(get_weather, location="blackpool")
print(get_function_info(gw, ignore_defaults=True))
print(get_non_default_signature(gw))
