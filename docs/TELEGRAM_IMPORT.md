# Импорт Telegram JSON export

Telegram Desktop умеет экспортировать историю в JSON. Этот проект использует такой JSON как offline-обучающую выборку для памяти и интересов Бямки.

## Быстрый запуск

```bash
python scripts/import_telegram_export.py \
  --json data/telegram_export/result.json \
  --target-chat-id 123456789 \
  --config config/config.yaml
```

`--target-chat-id` — это чат, на который должна повлиять история. Например, если ты хочешь, чтобы история повлияла на личный диалог с ботом, укажи id личного диалога. Если на группу — id группы.

## Full export с несколькими чатами

Если JSON содержит `chats.list`, нужно выбрать исходный чат:

```bash
python scripts/import_telegram_export.py \
  --json data/telegram_export/result.json \
  --source-chat-name "Название чата" \
  --target-chat-id 123456789
```

или точнее:

```bash
python scripts/import_telegram_export.py \
  --json data/telegram_export/result.json \
  --source-chat-id 123456789 \
  --target-chat-id 123456789
```

## Dry-run

Проверить, что файл читается и какой чат выбран:

```bash
python scripts/import_telegram_export.py \
  --json data/telegram_export/result.json \
  --target-chat-id 123456789 \
  --dry-run \
  --no-summaries \
  --no-facts \
  --no-interests
```

## Что именно импортируется

### 1. История сообщений

Сообщения сохраняются в `messages` с выбранным `target_chat_id`.

По умолчанию `message_id` из export преобразуется в отрицательный:

```yaml
telegram_import:
  message_id_mode: "negative_export_id"
```

Это снижает риск конфликта с реальными сообщениями Telegram в этом же чате.

### 2. Summaries

Скрипт режет историю на чанки и сохраняет compact summary в `chat_summaries`.

Summary должны хранить:

- стиль общения;
- важные факты;
- договорённости;
- локальные шутки;
- правила поведения;
- незавершённые темы.

### 3. Факты

Факты извлекаются батчами через LLM и сохраняются в SQLite + Qdrant.

Типы фактов:

- `chat_rule`;
- `user_preference`;
- `participant_preference`;
- `project_fact`;
- `decision`;
- `running_joke`;
- `local_topic`;
- `social_pattern`;
- `other`.

### 4. Правила поведения чата

Правила поведения сохраняются как факты с source:

```text
telegram_import:chat_rule
```

Потом они могут подтягиваться в behavior-memory и попадать в system prompt.

### 5. Learned interests

Каждое подходящее сообщение проигрывается через `PersonalityInterestEngine`.

Это обновляет:

- `interest_observations`;
- `interest_topics`;
- `learned_interest` для похожих тем.

Эти интересы локальны для `target_chat_id`.

### 6. Локальные темы и шутки

Они сохраняются двумя путями:

- как факты `running_joke` / `local_topic`;
- как темы в `interest_topics`, если replay интересов показал устойчивый интерес.

## Настройки

Секция конфига:

```yaml
telegram_import:
  import_messages: true
  build_summaries: true
  extract_facts: true
  update_interests: true

  message_id_mode: "negative_export_id"

  summary_chunk_size: 80
  summary_max_tokens: 900

  fact_chunk_size: 60
  max_facts_per_chunk: 12
  fact_min_confidence: 0.76
  fact_duplicate_score_threshold: 0.91

  interest_max_messages: 0
  progress_every_messages: 100
  progress_every_chunks: 5
```

`interest_max_messages: 0` означает обработать все подходящие сообщения. Если история очень большая, можно поставить, например, `3000`.

## Отключение отдельных этапов

```bash
--no-messages
--no-summaries
--no-facts
--no-interests
```

Например, только импортировать сообщения:

```bash
python scripts/import_telegram_export.py \
  --json result.json \
  --target-chat-id 123456789 \
  --no-summaries \
  --no-facts \
  --no-interests
```

## Итоговая статистика

После завершения скрипт печатает:

- сколько сообщений было в JSON;
- сколько текстовых сообщений удалось распарсить;
- сколько новых сообщений записано;
- сколько было дублей;
- сколько summary создано;
- сколько фактов сохранено;
- факты по типам;
- сколько сообщений прошло replay интересов;
- top learned interests;
- recent saved facts.

## Важные замечания

- Это не меняет веса LLM.
- Импорт может быть долгим: summaries, facts и interests используют Ollama.
- Для фактов нужен Qdrant local + embedding model.
- Для replay интересов нужен embedding model.
- Перед большим импортом полезно сделать `--dry-run`.
