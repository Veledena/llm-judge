import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from dialogues import parse_dialogue, to_lines
from judge import judge_reply
from metrics import REGISTRY


def read_dialogues(data_path: str):
    """Читает .jsonl: одна строка файла = один диалог."""
    with open(data_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            # Если в данных нет id диалога, используем номер строки
            yield row.get("dialogue_id", idx), row["string_agg"]

def build_tasks(data_path: str, metric_names: list[str]) -> list[tuple]:
    """Составляет список заданий: (диалог, номер реплики, метрика, контекст, реплика).

    Задание создаётся для КАЖДОЙ реплики оператора и КАЖДОЙ выбранной метрики.
    """
    tasks = []
    for dialogue_id, raw_text in read_dialogues(data_path):
        lines = to_lines(parse_dialogue(raw_text))
        for i, line in enumerate(lines):
            if line.startswith("[Оператор]"):
                for name in metric_names:
                    # lines[:i] = всё, что было ДО этой реплики
                    tasks.append((dialogue_id, i, name, lines[:i], line))
    return tasks

def load_done_keys(out_path: Path) -> set:
    """Какие задания уже успешно посчитаны (чтобы не считать заново после сбоя)."""
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["error"] is None:
                done.add((r["dialogue_id"], r["turn_idx"], r["metric"]))
    return done

def run_task(task: tuple) -> dict:
    """Выполняет одно задание. Эта функция запускается в потоках."""
    dialogue_id, turn_idx, metric_name, context, reply = task
    result = judge_reply(REGISTRY[metric_name], context, reply)
    return {"dialogue_id": dialogue_id, "turn_idx": turn_idx, "metric": metric_name,
            "reply": reply, **result}

def summarize(out_path: Path) -> pd.DataFrame:
    """Строит итоговую таблицу: среднее по диалогам и доля ошибок."""
    df = pd.read_json(out_path, lines=True)
    # Если задание считалось повторно, оставляем последнюю запись
    df = df.drop_duplicates(["dialogue_id", "turn_idx", "metric"], keep="last")

    rows = []
    for metric, group in df.groupby("metric"):
        valid = group[group["error"].isna() & group["label"].notna()]
        per_dialogue = valid.groupby("dialogue_id")["label"].mean()   # среднее по диалогу
        rows.append({
            "metric": metric,
            "score": per_dialogue.mean(),            # среднее по диалогам = итог метрики
            "n_dialogues": len(per_dialogue),
            "n_replies": len(valid),                 # реплик, вошедших в расчёт
            "error_rate": group["error"].notna().mean(),   # доля упавших запросов
        })
    return pd.DataFrame(rows)

def main():
    # Параметры из командной строки
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="путь к .jsonl с диалогами")
    parser.add_argument("--out", default="results/judge_results.jsonl")
    parser.add_argument("--metrics", nargs="+", default=list(REGISTRY),
                        help="какие метрики считать (по умолчанию все)")
    parser.add_argument("--workers", type=int, default=8,
                        help="сколько запросов к API одновременно")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Готовим задания и выкидываем уже посчитанные
    done = load_done_keys(out_path)
    tasks = [t for t in build_tasks(args.data, args.metrics)
             if (t[0], t[1], t[2]) not in done]
    print(f"Заданий к выполнению: {len(tasks)}")

    # 2. Выполняем параллельно. Результат пишем сразу, строка за строкой:
    #    если скрипт упадёт, уже посчитанное не пропадёт.
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
         open(out_path, "a", encoding="utf-8") as out:
        futures = [pool.submit(run_task, t) for t in tasks]
        for k, future in enumerate(as_completed(futures), start=1):
            out.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
            out.flush()
            if k % 50 == 0:
                print(f"Готово {k}/{len(tasks)}")

    # 3. Итоговая таблица
    summary = summarize(out_path)
    summary.to_csv(out_path.with_name("summary.csv"), index=False)
    print(summary)


if __name__ == "__main__":
    main()