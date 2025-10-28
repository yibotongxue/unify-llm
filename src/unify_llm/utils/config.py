from copy import deepcopy
from typing import Any

import yaml  # type: ignore [import-untyped]

__all__ = [
    "load_config",
    "update_config_with_unparsed_args",
]


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict[str, Any]: Loaded configuration.
    """
    with open(config_path) as file:
        config: dict[str, Any] = yaml.safe_load(file)
    return config


def update_dict(
    total_dict: dict[str, Any], item_dict: dict[str, Any]
) -> dict[str, Any]:

    def _is_list_dict(obj: Any) -> bool:
        return isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict)

    def _update_dict_from_top(
        total_dict: dict[str, Any], item_dict: dict[str, Any]
    ) -> dict[str, Any]:
        for key in total_dict.keys():
            if key in item_dict:
                if isinstance(item_dict[key], dict) and isinstance(
                    total_dict[key], dict
                ):
                    _update_dict_from_top(total_dict[key], item_dict[key])
                elif isinstance(item_dict[key], list) and _is_list_dict(
                    total_dict[key]
                ):
                    for element in total_dict[key]:
                        _update_dict(element, item_dict[key])
                else:
                    total_dict[key] = item_dict[key]
        return total_dict

    def _update_dict(
        total_dict: dict[str, Any], item_dict: dict[str, Any]
    ) -> dict[str, Any]:

        for key, value in total_dict.items():
            if key in item_dict:
                if isinstance(item_dict[key], dict) and isinstance(
                    total_dict[key], dict
                ):
                    _update_dict(total_dict[key], item_dict[key])
                elif isinstance(item_dict[key], dict) and _is_list_dict(
                    total_dict[key]
                ):
                    for element in total_dict[key]:
                        _update_dict(element, item_dict[key])
                else:
                    total_dict[key] = item_dict[key]
            if isinstance(value, dict):
                _update_dict(value, item_dict)
            elif _is_list_dict(total_dict[key]):
                for element in value:
                    _update_dict(element, item_dict)
        return total_dict

    if "" in item_dict:
        item_dict_from_top = item_dict.pop("")
        if isinstance(item_dict_from_top, dict):
            total_dict = _update_dict_from_top(total_dict, item_dict_from_top)

    return _update_dict(total_dict, item_dict)


def is_convertible_to_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def custom_cfgs_to_dict(key_list: str, value: Any) -> dict[str, Any]:
    """This function is used to convert the custom configurations to dict."""
    if value == "True":
        value = True
    elif value == "False":
        value = False
    elif value == "None" or value == "null":
        value = None
    elif value.isdigit():
        value = int(value)
    elif is_convertible_to_float(value):
        value = float(value)
    elif value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
        value = value.split(",")
        value = list(filter(None, value))
    elif "," in value:
        value = value.split(",")
        value = list(filter(None, value))
    else:
        value = str(value)
    keys_split = key_list.replace("-", "_").split(":")
    return_dict = {keys_split[-1]: value}

    for key in reversed(keys_split[:-1]):
        return_dict = {key.replace("-", "_"): return_dict}
    return return_dict


def update_config_with_unparsed_args(
    unparsed_args: list[str], cfgs: dict[str, Any]
) -> None:
    keys = [k[2:] for k in unparsed_args[::2]]
    values = list(unparsed_args[1::2])
    unparsed_args_dict = dict(zip(keys, values))

    for k, v in unparsed_args_dict.items():
        cfgs = update_dict(cfgs, custom_cfgs_to_dict(k, v))


def deepcopy_config(cfgs: dict[str, Any]) -> dict[str, Any]:

    def _deepcopy_config(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: _deepcopy_config(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_deepcopy_config(e) for e in obj]
        return deepcopy(obj)

    return _deepcopy_config(cfgs)  # type: ignore [return-value]
