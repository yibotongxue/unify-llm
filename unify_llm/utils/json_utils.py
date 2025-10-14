import json
import os
from typing import Any


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """
    Load a JSON Lines file and return a list of dictionaries.

    Args:
        file_path (str): The path to the JSON Lines file.

    Returns:
        list[dict]: A list of dictionaries loaded from the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    data = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def load_json(file_path: str) -> dict[str, Any]:
    """
    Load a JSON file and return its content as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The content of the JSON file as a dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, encoding="utf-8") as file:
        return json.load(file)  # type: ignore [no-any-return]


def save_json(data: dict[str, Any], file_path: str) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to save.
        file_path (str): The path where the JSON file will be saved.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def save_jsonl(data: list[dict[str, Any]], file_path: str) -> None:
    """
    Save a list of dictionaries to a JSON Lines file.

    Args:
        data (list[dict]): The list of dictionaries to save.
        file_path (str): The path where the JSON Lines file will be saved.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
