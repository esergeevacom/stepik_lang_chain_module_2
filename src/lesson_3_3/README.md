# RAG preprocessing pipeline

`preprosess.py` загружает записи через `datasets.load_dataset`, очищает текст, добавляет базовые метаданные, удаляет короткие и дублирующиеся записи, а затем сохраняет результат в `jsonl`.

## Что делает скрипт

- загружает датасет из Hugging Face
- берет текст из одного из полей: `text`, `context`, `article`, `abstract`, `page_content`
- нормализует пробелы
- добавляет метаданные: `dataset_name`, `dataset_split`, `text_length`, `content_hash`
- копирует базовые поля метаданных, если они есть: `id`, `title`, `url`, `source`, `page`, `chunk_id`, `chunk_index`
- фильтрует короткие тексты
- удаляет полные дубликаты по хэшу текста
- сохраняет результат в `jsonl`

## Зависимости

- Python 3.13
- `datasets`
- `langchain-core`
- `pydantic`

## Пример запуска

Пример запуска препроцессинга на примере датасета русской Википедии:

```bash
python preprosess.py \
  --dataset-name wikimedia/wikipedia \
  --dataset-config ru \
  --split "train[:20]" \
  --min-text-length 200 \
  --remove-duplicates \
  --output prepared.jsonl
```

## Формат выхода

Каждая строка в `jsonl` содержит объект такого вида:

```json
{"page_content": "...", "metadata": {"dataset_name": "wikimedia/wikipedia", "dataset_split": "train[:20]", "text_length": 1234, "content_hash": "..."}}
```
