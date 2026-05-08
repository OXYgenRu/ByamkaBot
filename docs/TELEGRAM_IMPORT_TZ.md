# ТЗ: offline-импорт Telegram JSON истории в память Бямки

## Цель

Добавить механизм, который принимает JSON export из Telegram Desktop и превращает историю чата в долговременное состояние бота для выбранного `target_chat_id`.

Это не fine-tuning модели. Скрипт должен менять окружение бота:

1. историю сообщений в SQLite;
2. summaries старых фрагментов;
3. факты в SQLite + Qdrant;
4. правила поведения чата;
5. learned interests в `interest_topics`;
6. локальные темы и шутки.

## Основные требования

- Пользователь явно задаёт `--target-chat-id`, на какой чат должна повлиять импортируемая история.
- Поддерживаются два формата Telegram Desktop export:
  - single chat export: корневой объект содержит `messages`;
  - full account export: корневой объект содержит `chats.list`.
- Для full export можно выбрать исходный чат через `--source-chat-id` или `--source-chat-name`.
- Процесс должен показывать прогресс, потому что импорт может занимать долгое время.
- Скрипт должен печатать итоговую статистику: сколько сообщений прочитано, импортировано, пропущено, сколько summary/facts/interests создано.
- Скрипт должен показывать, какие интересы бот выучил после replay.
- Все настройки импорта должны быть в YAML-конфиге.
- Повторный импорт не должен массово дублировать сообщения и факты.

## Архитектурное решение

Добавлены:

- `bot_app/importers/telegram_export.py` — основной importer;
- `scripts/import_telegram_export.py` — CLI-скрипт;
- секция `telegram_import` в `config/config.example.yaml`;
- дополнительные методы статистики в `SQLiteStore`.

## Pipeline

```text
Telegram result.json
  ↓
parse/select chat
  ↓
normalize messages
  ↓
SQLite messages for target_chat_id
  ↓
batch summaries
  ↓
chunk fact extraction
  ↓
Qdrant + SQLite facts
  ↓
interest replay
  ↓
final import statistics
```

## Безопасность контекста

Импорт всегда пишет в конкретный `target_chat_id`. Автоматически глобальные факты не создаются.

По умолчанию экспортные `message_id` сохраняются как отрицательные (`negative_export_id`), чтобы не конфликтовать с реальными Telegram-сообщениями чата.
