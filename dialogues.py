import re

ROLE_TAG = {"CLIENT": "[Клиент]", "OPERATOR": "[Оператор]"}

def preprocess_greenplum_text(raw_text: str):
    parts = re.split(r"(CLIENT:|OPERATOR:)", raw_text)

    turns = []          # итоговый список реплик
    current_role = None  # кто сейчас говорит

    for part in parts:
        part = part.strip()
        if part in ("CLIENT:", "OPERATOR:"):
            current_role = part[:-1]          # отрезаем двоеточие
        elif part and current_role:
            # Если подряд две реплики одного человека, склеиваем их в одну
            if turns and turns[-1][0] == current_role:
                turns[-1] = (current_role, turns[-1][1] + " " + part)
            else:
                turns.append((current_role, part))
    return turns

def to_lines(turns: list[tuple[str, str]]) -> list[str]:
    """[("CLIENT", "Привет")] -> ["[Клиент]: Привет"]"""
    return [f"{ROLE_TAG[role]}: {text}" for role, text in turns]