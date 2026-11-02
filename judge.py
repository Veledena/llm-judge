import json
import re
import time

from gigachat_api import gc
from metrics import Metric

SYSTEM_PROMPT = """Ты — судья, оценивающий реплику оператора банка в голосовом диалоге.
Критерий оценки: {name}.

{descrip}

Отвечай строго JSON без текста вокруг, в таком формате:
{{"reasoning": "<одно-два предложения>", "label": 0, 1 или null}}"""


def format_query(context: list[str], phrase: str) -> str:
    """ Единый формат запроса к судье.
    Одна и та же функция используется для few-short примеров и 
    последующих запросов, чтобы форматы не разошлись, а ллмка не путалась, а то я запуталась уже)
    """
    context_text = "\n".join(context) if context else "(начало диалога)"
    return (
        f"КОНТЕКСТ ДИАЛОГА:\n{context_text}\n\n"
        f"ОЦЕНИВАЕМАЯ РЕПЛИКА ОПЕРАТОРА:\n{phrase}"
    )

def build_system_prompt(metric: Metric) -> str:
    """Системный промтЖ роль судьи + конкретная метрика"""
    return SYSTEM_PROMPT.format(name=metric.name, descrip=metric.descrip)

def build_few_shot(metric: Metric) -> list[dict]:
    """Превращает примеры метрики в «фальшивую историю чата».

    Каждый пример = пара сообщений:
      user      -> запрос в том же формате, что и боевой
      assistant -> идеальный ответ судьи (JSON)
    Этот список потом передаётся в параметр chat_history функции gc().
    """
    history = []
    for ex in metric.examples:
        history.append({"role": "user", "content": format_query(list(ex.context), ex.reply)})
        history.append({
            "role": "assistant",
            # reasoning стоит ПЕРЕД label: модель сначала «рассуждает», потом решает
            "content": json.dumps({"reasoning": ex.reasoning, "label": ex.label},
                                  ensure_ascii=False),
        })
    return history


def parse_judge_answer(text: str):
    """Достаёт label и reasoning из ответа модели.

    Модель иногда оборачивает JSON в ```json ... ``` или пишет текст вокруг,
    поэтому ищем первую пару фигурных скобок, а не парсим всю строку.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ValueError(f"В ответе нет JSON: {text[:200]}")
    answer = json.loads(match.group(0))

    label = answer["label"]
    if label is not None:
        label = int(label)
        if label not in (0, 1):
            raise ValueError(f"Недопустимый label: {label}")
    return label, str(answer.get("reasoning", ""))


def judge_reply(metric: Metric, context: list[str], reply: str, max_retries: int = 3) -> dict:
    """Оценивает одну реплику оператора по одной метрике.

    context  — все реплики ДО оцениваемой (полный диалог с начала)
    reply    — сама реплика оператора
    Возвращает словарь: label, reasoning, error.
    """
    # 1. ОБРЕЗАЕМ КОНТЕКСТ ПОД МЕТРИКУ. Здесь работает context_window:
    #    context[-4:] берёт последние 4 реплики; None -> берём всё.
    window = metric.context_window
    visible_context = context[-window:] if window else context

    # 2. Собираем три части запроса
    query = format_query(visible_context, reply)       # что оцениваем
    system_prompt = build_system_prompt(metric)        # инструкция судье
    few_shot = build_few_shot(metric)                  # примеры

    # 3. Отправляем. При сбое (сеть, кривой JSON) повторяем с паузой 1с, 2с, 4с.
    last_error = None
    for attempt in range(max_retries):
        try:
            messages, _ = gc(
                query=query,
                system_prompt=system_prompt,   # <- инструкции идут сюда
                chat_history=few_shot,         # <- примеры идут сюда
                temperature=0.0,               # 0 = максимально стабильные ответы
            )
            label, reasoning = parse_judge_answer(messages[-1]["content"])
            return {"label": label, "reasoning": reasoning, "error": None}
        except Exception as e:
            last_error = repr(e)
            time.sleep(2 ** attempt)

    # 4. Все попытки провалились: label = None и текст ошибки.
    #    Никаких «-100» в среднее не подставляем, ошибки считаем отдельно.
    return {"label": None, "reasoning": None, "error": last_error}

