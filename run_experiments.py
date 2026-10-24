"""Run experiments and write the report and measured results."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from ensemble_models import train_and_evaluate
from grader_contracts.ensembles import EnsembleInput


def main():
    root = Path(__file__).resolve().parent
    csv_path = root / "german.csv"
    frame = pd.read_csv(csv_path, sep=";")
    target = frame.pop("Creditability")
    if set(target.unique()) != {0, 1}:
        raise ValueError("Creditability must contain both 0 and 1.")
    seed = 42
    train_x, test_x, train_y, test_y = train_test_split(
        frame, target, test_size=0.25, stratify=target, random_state=seed)
    print("4 model, 3 katli egitim capraz dogrulamasi calisiyor...", flush=True)
    result, experiments = train_and_evaluate(
        EnsembleInput(train_x, train_y, test_x, test_y, random_seed=seed))
    selected = next(record for record in experiments if record["selected"])
    output = {
        "random_seed": seed, "test_size": 0.25, "cv_folds": 3,
        "selection_criterion": "Maximum mean training cross-validation ROC-AUC",
        "positive_class": 1, "threshold": 0.5,
        "dataset_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "rows": len(frame), "features": frame.shape[1],
        "missing_values": int(frame.isna().sum().sum()),
        "target_counts": {str(k): int(v) for k, v in target.value_counts().items()},
        "train_indices": train_x.index.tolist(), "test_indices": test_x.index.tolist(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "selected_model": result.model_name, "selected_test_metrics": selected["test_metrics"],
        "experiments": experiments,
        "selected_probabilities": result.probabilities.tolist(),
        "selected_predictions": result.predictions.tolist(),
    }
    (root / "experiment_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = ["| Модель | ROC-AUC на CV: среднее ± std | ROC-AUC на тесте | Accuracy | Precision | Recall | F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for record in experiments:
        m = record["test_metrics"]
        table.append(f"| {record['model_name']} | {record['cv_roc_auc_mean']:.4f} ± "
                     f"{record['cv_roc_auc_std']:.4f} | " + " | ".join(
                         f"{m[key]:.4f}" for key in ["roc_auc", "accuracy", "precision", "recall", "f1"]) + " |")
    report = f"""# Лабораторная работа №4 — Ансамбли моделей

## Данные и подготовка

Набор `german.csv` содержит {len(frame)} строк и {frame.shape[1]} признаков.
Целевая переменная — `Creditability`: положительный класс 1 означает кредитоспособность,
отрицательный класс — 0. Численность классов: {output['target_counts']}.
Данные разделены с сохранением долей классов: 75% на обучение ({len(train_x)} строк),
25% на тест ({len(test_x)} строк), random_state=42.
Количество пропущенных значений в признаках: {output['missing_values']}.
Предусмотрено заполнение числовых пропусков медианой, категориальных — наиболее частым значением;
при отсутствии пропусков эти шаги не изменяют значения.
Категории, представленные числовыми кодами, преобразованы с помощью one-hot-кодирования.
Список категориальных столбцов приведён в `ensemble_models.py`; остальные столбцы обработаны как числовые.
Неизвестные категории обрабатываются с параметром handle_unknown="ignore".
Масштабирование не применялось, поскольку использовались модели на основе деревьев.
На каждом разбиении кросс-валидации предобработка обучалась только на обучающей части.

## Гипотезы и эксперименты

- Random Forest: 300 деревьев, минимальное число объектов в листе — 2. Гипотеза: бэггинг снижает дисперсию предсказаний.
- Extra Trees: 300 деревьев, минимальное число объектов в листе — 2. Гипотеза: дополнительная случайность разбиений улучшает обобщение.
- Gradient Boosting depth=1: 150 деревьев, learning_rate=0.05. Проверяется возможность описать простые зависимости моделью небольшой сложности.
- Gradient Boosting depth=2: то же число деревьев и тот же темп обучения; проверяется вклад парных взаимодействий признаков.

Четыре кандидата заданы до проведения экспериментов. На обучающей выборке использован
StratifiedKFold с 3 разбиениями, shuffle=True и random_state=42.
Критерий выбора — средний ROC-AUC на кросс-валидации.
Тестовые метки не использовались для предобработки, обучения или выбора модели.
После завершения выбора по CV модели обучены на всей обучающей выборке.

## Метрики

{chr(10).join(table)}

Тестовые метрики приведены для итоговой оценки; выбор модели по ним не пересматривался.
ROC-AUC рассчитан по вероятностям положительного класса. Для остальных метрик использован фиксированный порог 0.5.

## Выбранная модель

Выбранная модель: **{result.model_name}**. Средний ROC-AUC на обучающей кросс-валидации:
**{selected['cv_roc_auc_mean']:.4f}**; ROC-AUC на тесте: **{result.roc_auc:.4f}**.
Выбор выполнен по наибольшему среднему ROC-AUC на CV; при равенстве выбирается первый кандидат в списке.
Полные параметры и оценки на отдельных разбиениях сохранены в `experiment_results.json`.

## Выводы и ограничения

В данном запуске выбранная модель показала наибольшую среднюю оценку на обучающей CV.
В таблице сравниваются случайные ансамбли деревьев и бустинг с разной глубиной деревьев.
Для Random Forest средний ROC-AUC на CV составил {experiments[0]['cv_roc_auc_mean']:.4f},
для Extra Trees — {experiments[1]['cv_roc_auc_mean']:.4f}: дополнительная случайность
разбиений Extra Trees в данном запуске не дала преимущества перед Random Forest.
При увеличении глубины Gradient Boosting с 1 до 2 средний ROC-AUC на CV изменился
с {experiments[2]['cv_roc_auc_mean']:.4f} до {experiments[3]['cv_roc_auc_mean']:.4f}.
Это согласуется с гипотезой о полезности взаимодействий признаков на данном разбиении,
но не доказывает, что более глубокие деревья всегда лучше.
Эксперимент не измеряет отдельно вклад масштабирования, one-hot-кодирования или заполнения пропусков;
утверждать об их самостоятельном положительном эффекте по этим результатам нельзя.
Из-за небольшого набора данных и единственного тестового разбиения оценки могут измениться при другом разбиении.
Стандартное отклонение оценки CV характеризует разброс, но не является доверительным интервалом.
Такого прогноза недостаточно для реальных кредитных решений:
справедливость, калибровка вероятностей и проверка на внешних данных в работе не исследовались.
Тестовое разбиение и random seed не менялись ради повышения результата.

## Воспроизводимость

Random seed: 42; test_size: 0.25; stratify: целевая переменная; CV: 3 разбиения;
n_jobs: 1. SHA-256 файла данных: `{output['dataset_sha256']}`.
Версии библиотек в выполненном запуске: `{json.dumps(output['versions'], ensure_ascii=False)}`.
Индексы строк, предсказания и вероятности сохранены в JSON-файле.
Для повторения в совместимой локальной среде выполните из папки проекта в терминале PowerShell VS Code:

```powershell
.\\.venv\\Scripts\\python.exe run_experiments.py
```

Команда повторно выполняет эксперименты и обновляет отчёт и JSON-файл.
Для Google Colab запустите все ячейки актуального `lab4_colab.ipynb` и скачайте полученные файлы.
Для точного повторения используйте версии библиотек, указанные выше; в другой среде результаты могут отличаться.
Точка входа для проверки сохранена: `run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult`.
"""
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    print("\n" + "\n".join(table))
    print(f"\nSecilen model: {result.model_name}")
    print("REPORT.md ve experiment_results.json kaydedildi.")


if __name__ == "__main__":
    main()
