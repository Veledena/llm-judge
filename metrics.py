from dataclasses import dataclass

# dataclass - это для "__init__" (инициализации экземпляра), "__repr__" (красивый вывод в консоль), 
# "__eq__" (метод сравнения) из коробки
# декоратор @dataclass генерирует все эти методы автоматически

@dataclass(frozen=True) # чтобы объект класса был неизменяемым
class Example:
    context: tuple       # реплики до оцениваемой, например ("[Клиент]: Дорого",)
    phrase: str         # сама реплика для оценки
    reasoning: str      # краткое объяснение оценки
    label: int | None   # 1, 0, None (метрика не применима к этой реплике)

@dataclass(frozen=True)
class Metric:
    name: str                   # название метрики
    descrip: str                # что значит 1, что значит 0
    examples: tuple             # few-shot примеры
    context_window: int | None # сколько реплик до видит судья
                                # (None = весь диалог с начала)

# -------- Метрика 1 ----------
NATURALNESS = Metric(
    name="naturalness",
    context_window=4,
    descrip=(
        "label=1: реплика звучит живо, по-человечески, как в телефонном разговоре\n"
        "label=0: реплика шаблонная, роботизированная или перегружена канцеляризмами."
    ),
    exaples=(
        Example(
            context=("...","..."),
            phrase="...",
            reasoning="...",
            label=1),
        Example(...

        )))

# -------- Метрика 2 ----------

....

# Реестр: по имени из командной строки находим нужную метрику. ??
# Новая метрика = новый блок Metric(...) выше + одна запись здесь.
REGISTRY = {m.name: m for m in (NATURALNESS)}
