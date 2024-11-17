import inspect
import typing
from functools import partial
from inspect import Parameter, Signature
from typing import Any, Callable

from pydantic import create_model

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
            if param.default == Parameter.empty and name not in func.keywords.keys()
        ]
    params = inspect.signature(func).parameters
    return [name for name, param in params.items() if param.default == Parameter.empty]


def function_schema(func: Callable) -> dict:
    kw = {
        n: (o.annotation, ... if o.default == Parameter.empty else o.default)
        for n, o in inspect.signature(func).parameters.items()
    }
    s = create_model(f"Input for `{func.__name__}`", **kw).schema()  # type: ignore
    return dict(name=func.__name__, description=func.__doc__, parameters=s)


def skip_default_params_(func: Callable) -> Signature:
    if isinstance(func, partial):
        signature = inspect.signature(func.func)
        print(f"Keywords: {func.keywords}")
        return signature.replace(
            parameters=[
                p
                for p in signature.parameters.values()
                if p.default == Parameter.empty and p.name not in func.keywords
            ]
        )
    else:
        signature = inspect.signature(func)
        return signature.replace(
            parameters=[p for p in signature.parameters.values() if p.default == Parameter.empty]
        )


def get_function_signature(
    func: Callable, ignore_default_values: bool = False, skip_default_params: bool = False
) -> Signature:
    if skip_default_params:
        return skip_default_params_(func)
    else:
        if isinstance(func, partial):
            signature = inspect.signature(func.func)
        else:
            signature = inspect.signature(func)
        if ignore_default_values:
            return signature.replace(
                parameters=[
                    Parameter(name=p.name, kind=p.kind, default=Parameter.empty, annotation=p.annotation)
                    for p in signature.parameters.values()
                ]
            )
        else:
            return signature


def function_to_type(
    func: Callable | type, ignore_default_values: bool = False, skip_default_params: bool = False
) -> type:
    if isinstance(func, type):
        return func
    kw = {
        n: (
            str if o.annotation == Parameter.empty else o.annotation,
            ... if (o.default == Parameter.empty or ignore_default_values) else o.default,
        )
        for n, o in (
            get_function_signature(
                func, ignore_default_values=ignore_default_values, skip_default_params=skip_default_params
            )
        ).parameters.items()
    }
    if isinstance(func, partial):
        func = func.func
    return create_model(func.__name__, __doc__=inspect.getdoc(func), **kw)  # type:ignore


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


def get_function_info(
    func: Callable, ignore_default_values: bool = False, skip_default_params: bool = False
) -> dict[str, str]:
    signature = get_function_signature(
        func=func, ignore_default_values=ignore_default_values, skip_default_params=skip_default_params
    )
    func = func.func if isinstance(func, partial) else func
    name = func.__name__
    docstring = inspect.getdoc(func)
    info = {"name": name}
    if signature:
        info["signature"] = f"{name}{signature}"
    if docstring:
        info["docstring"] = deindent(docstring)
    return info


def create_function_info(
    name: str, parameters: dict[str, tuple[type, Any]] | None = None, docstring: str = ""
) -> dict[str, str]:
    parameters = parameters or {}
    sig_params = []
    for param_name, (param_type, default_value) in parameters.items():
        sig_params.append(
            Parameter(
                name=param_name,
                annotation=param_type,
                default=Parameter.empty if default_value is ... else default_value,
                kind=Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
    signature = Signature(parameters=sig_params)
    info = {"name": name}
    info["signature"] = f"{name}{signature}".replace("__main__.", "")
    if docstring:
        info["docstring"] = docstring
    return info


def type_to_function_info(typ: type, func_name: str, param_name: str, docstring: str) -> dict[str, str]:
    info = {"name": func_name}
    info["signature"] = f"{func_name}({param_name}: {typ.__name__})"
    if docstring:
        info["docstring"] = docstring
    return info


