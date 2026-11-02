import time
import requests
import os
import json
from typing import Optional, List, Dict 
from functools import wraps

from dotenv import load_dotenv
load_dotenv(override=True)

GIGACHAT_API_URL = "http://liveaccess/v1/gc"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('JPY_API_TOKEN')}",
    "Content-Type": "application/json",
}

def timer(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        st = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_time = time.perf_counter() - st 
        return result, elapsed_time 
    return wrapper

def models():
    response = requests.get(
        url = GIGACHAT_API_URL + "/models",
        headers = HEADERS,
    )

    if response.ok:
        return json.dumps(response.json(), indent=4, ensure_ascii=False)
    else:
        return response.text
    
def completions(
        query: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
):
    model = model if model else "GigaChat-3-Ultra"
    system_prompt_prefix = [] if system_prompt is None else [{ "role": "system", "content": system_prompt}]
    messages = []

    if chat_history is not None:
        if not isinstance(chat_history, list):
            raise ValueError(f"FATAL: `chat_history` must be `None` or a list of `dict`s. Got {type(chat_history)} instead.")
        for el in chat_history:
            if not isinstance(el, dict):
                raise ValueError(f"FATAL: each element of `chat_history must be a `dict`. Got {type(el)} instead.")
            if not "role" in el or not "content" in el:
                raise ValueError(f"FATAL: each element of `chat_history` must contain the keys `role` and `content`.")
            if el["role"] == 'system':
                raise ValueError(f"FATAL: no element of `chat_history` can contain `role` == `system`.")
        messages = [*messages, *chat_history]

    messages.append(
        {"role": "user",
         "content": query,
         })
    data = {
        "model": model,
        "messages": [*system_prompt_prefix, *messages],
        "n": 1,
        "temperature": temperature if temperature else 0.
    }
    response = requests.post(
        url=GIGACHAT_API_URL + "/chat/completions",
        headers=HEADERS,
        json=data,
    )
    if response.ok:
        messages.append(parse_response_into_json(json.dumps(response.json(), indent=4, ensure_ascii=False)))
    else:
        messages.append(parse_response_into_json(response.text))
    return messages

def parse_response_into_json(response: str):
    try:
        choices = json.loads(response).get("choices", [])
    except Exception as _:
        raise Exception(f"FATAl: Failed to parse choices from `response`:\n\n{response}")
    
    if len(choices) < 1:
        raise ValueError(f"FATAL: Parsed `choices` are of len {len(choices)}. Minimum of 1 is expected. \n`choices`:\n{choices}\n`response`:\n{response}")
    
    if not isinstance(choices[-1], dict):
        raise ValueError(f'FATAL: The elements of `choices` must be of type `dict`. Got {type(choices[-1])} instead.')
    
    last_message = choices[-1].get("message", None)

    if not isinstance(last_message, dict):
        raise ValueError(f"FATAL: `last_message` must be of type `dict`. Got {type(last_message)} instead.")
    
    content = last_message.get("content", None)
    if not isinstance(content, str):
        raise ValueError(f"FATAL: `content` must be of type `str`. Got {type(content)} instead.")
    
    role = last_message.get("role", None)
    if not isinstance(role, str):
        raise ValueError(f"FATAL: `role` must be of type `str`. Got {type(role)} instead.")
    
    return {
        "role": role,
        "content": content,
    }

@timer
def gc(
    query: str,
    system_prompt: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
):
    """
    Available models:
    - GigaChat-2-Max
    - GigaChat-2-Pro
    - GigaChat-3-Ultra <-- the default model
    """
    
    if not isinstance(query, str):
        raise ValueError("request must be a strng")
    
    return completions(query, system_prompt, chat_history, temperature, model) 