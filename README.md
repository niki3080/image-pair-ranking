# Image Pair Ranking

Проект подготовлен в рамках соревнования Kaggle [TETA NN 2 2026](https://www.kaggle.com/competitions/teta-nn-2-2026).

Задача: по паре сгенерированных изображений предсказать, какое изображение выбрала экспертная комиссия. Метрика соревнования: `ROC AUC`.

- Kaggle name: **Yulia Nikitina**
- Real name: **Юлия Никитина**
- Leaderboard: **2 место**

## Отчет

- [Основной ноутбук](img_processor.ipynb)
- [HTML export](https://niki3080.github.io/image-pair-ranking/img_processor.html)
- [Финальный сабмит](submissions/final_weighted_blend_submission.csv)

## Данные

Данные доступны на вкладке [Data](https://www.kaggle.com/competitions/teta-nn-2-2026/data) соревнования. Файлы нужно положить локально в папку `data/`.

Ожидаемые файлы:

| Файл | Описание |
|---|---|
| `train.csv` | обучающая выборка |
| `test.csv` | тестовая выборка |
| `sample_submission.csv` | пример сабмита в правильном формате |

Основные колонки `train.csv`:

| Колонка | Описание |
|---|---|
| `image_1` | изображение 1, сохраненное в бинарном виде |
| `image_2` | изображение 2, сохраненное в бинарном виде |
| `is_image1_better` | бинарная целевая переменная: выбрала ли комиссия `image_1` |

## Архитектура финального решения

```text
PIPELINE
│
├── предсказания базовых моделей
│   │
│   ├── CLIP ViT Base        | CatBoost | Siamese
│   ├── CLIP ViT Large       | CatBoost | Siamese
│   │
│   ├── SigLIP 256           | CatBoost | Siamese
│   ├── SigLIP 384           | CatBoost
│   │
│   ├── OpenCLIP ConvNeXt    | CatBoost
│   │
│   ├── EVA-CLIP Large       | CatBoost
│   └── EVA-CLIP Base        | CatBoost
│
├── отбор моделей: ablation test по OOF ROC AUC
│
└── стабилизация весов
    │
    ├── random search весов по OOF-предсказаниям
    ├── усреднение top-100 наборов весов
    └── final prediction: взвешенная сумма test-предсказаний выбранных моделей
```

Финальный ансамбль построен на pretrained visual embeddings, CatBoost-моделях, Siamese comparator и стабилизированном подборе весов по OOF-предсказаниям.

## Что есть в ноутбуке

- обзор данных и примеры пар изображений;
- baseline на ручных image statistics;
- извлечение визуальных эмбеддингов;
- обучение CatBoost на pairwise embedding features;
- обучение Siamese comparator;
- ablation test для отбора моделей;
- random search весов ансамбля;
- анализ ошибок;
- формирование финального submission-файла.

## Структура репозитория

```text
.
├── img_processor.ipynb    # основной ноутбук с кодом и отчетом
├── img_processor.html     # HTML-экспорт ноутбука
├── files/                 # картинки и таблицы для отчета
├── submissions/           # финальный submission
├── pyproject.toml
├── uv.lock
└── README.md
```

## Запуск

Установить окружение:

```bash
uv sync
```

После скачивания данных положить файлы в папку `data/` и открыть ноутбук:

```bash
jupyter lab img_processor.ipynb
```

Крупные локальные артефакты обучения и эмбеддингов не коммитятся:

- `data/`
- `artefacts/`
- `catboost_info/`
- `.venv/`
