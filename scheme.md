**Структура файлов для проверки**

llm_judge/
├── gigachat_api.py   # обращение к GigaChat (твой gc_api.ipynb, превращённый в .py)
├── metrics.py        # описание метрик: рубрика, few-shot примеры, размер контекста
├── dialogues.py      # превращение сырого текста диалога в список реплик
├── judge.py          # вызов судьи: собрать запрос, отправить, разобрать ответ
├── run_eval.py       # главный скрипт: прогон по датасету, запись, итоговая таблица
├── json_dialogues/test.jsonl   # входные данные
└── results/                    # сюда пишутся результаты

Как они связаны: **run_eval.py** читает данные, использует **dialogues.py** для разбора, **metrics.py** для настроек метрик и **judge.py** для самой оценки, которая внутри вызывает **gigachat_api.py**.

## 1. gigachat_api.py

Берёшь код из gc_api.ipynb:

- Удаляешь тестовые ячейки (requests.post(... embeddings), messages = None и вызов gc("вот тут писать запрос")). В модуле должны остаться только импорты, константы и функции.
- Сохраняешь как gigachat_api.py.
- Функцию gc(query, system_prompt, chat_history, temperature, model) не меняем. Она возвращает (messages, время), где messages[-1]["content"] — ответ модели.

## 2. metrics.py: метрики

Здесь классы хранят данные и не содержат логики. Это «паспорт» метрики.

## 3. dialogues.py: разбор сырого текста

## 4. judge.py: сам судья

## 5. run_eval.py: главный скрипт

Запуск: python run_eval.py --data json_dialogues/test.jsonl --metrics naturalness objection_handling --workers 8
