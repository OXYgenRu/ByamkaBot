"""Append complete example dialogues without rewriting existing examples."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EXAMPLES = Path(__file__).resolve().parents[1] / "config" / "dialogue_examples.yaml"
TELEGRAM_HEADER = re.compile(
    r"^\[\d{2}\.\d{2}\.\d{4} \d{1,2}:\d{2}(?::\d{2})?\] "
    r"(?P<author>.+?)(?: in reply to (?P<reply_to>.+?))?:\s*(?P<first>.*)$"
)


def validate_dialogue(dialogue: Any) -> list[dict[str, str]]:
    if not isinstance(dialogue, list) or not dialogue:
        raise ValueError("Диалог должен содержать хотя бы одну пару user/assistant")
    if len(dialogue) % 2:
        raise ValueError("Диалог заканчивается репликой user без ответа assistant. Добавь ответ бота или убери последнюю реплику")
    for index, message in enumerate(dialogue):
        role = "user" if index % 2 == 0 else "assistant"
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] != role or not isinstance(message["content"], str)
                or not message["content"].strip()):
            raise ValueError(f"Реплика {index + 1}: ожидается непустая роль {role} и поле content")
    return dialogue


def read_input(path: Path) -> list[list[dict[str, str]]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        if set(data) == {"dialogue"}:
            data = [data["dialogue"]]
        elif set(data) == {"dialogues"}:
            data = data["dialogues"]
        else:
            raise ValueError("Входной файл должен содержать dialogue или dialogues")
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        data = [data]
    if not isinstance(data, list) or not data:
        raise ValueError("Во входном файле нет диалогов")
    return [validate_dialogue(dialogue) for dialogue in data]


def read_telegram_text(path: Path, replace_speaker: str) -> list[list[dict[str, str]]]:
    """Turn one copied Telegram transcript into a complete example dialogue."""
    if not replace_speaker.strip():
        raise ValueError("Имя заменяемого участника не может быть пустым")
    posts: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    body: list[str] = []

    def finish() -> None:
        if current is None:
            return
        lines = [line[:-1] if line.endswith("\\") else line for line in body]
        own: list[str] = []
        quoted: list[str] = []
        in_quote = False
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith((">", "\\>")):
                in_quote = True
                quoted.append(stripped.lstrip("\\>").strip())
            elif in_quote and not stripped:
                quoted.append("")
            else:
                in_quote = False
                own.append(line)
        current["text"] = "\n".join(own).strip()
        if quoted:
            current["quoted_text"] = "\n".join(quoted).strip()
        if not current["text"]:
            raise ValueError(f"Пустая реплика участника {current['author']}")
        posts.append(current)

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = TELEGRAM_HEADER.match(line)
        if match:
            finish()
            current = {"author": match["author"].strip()}
            if match["reply_to"]:
                current["reply_to"] = match["reply_to"].strip()
            body = [match["first"]]
        else:
            if current is None:
                if line.strip():
                    raise ValueError("Файл должен начинаться строкой вида [22.09.2025 13:36] Имя: текст")
                continue
            body.append(line)
    finish()
    if not posts:
        raise ValueError("Во фрагменте нет сообщений Telegram")
    if not any(post["author"] == replace_speaker for post in posts):
        raise ValueError(f"Участник {replace_speaker!r} не найден во фрагменте")

    dialogue: list[dict[str, str]] = []
    group: list[dict[str, str]] = []
    group_is_bot: bool | None = None

    def add_group() -> None:
        if not group:
            return
        if group_is_bot:
            if not dialogue:
                dialogue.append({"role": "user", "content": json.dumps({"example_context":
                    "Начало независимого примера. Я публикую сообщение в чате без внешнего обращения."},
                    ensure_ascii=False)})
            dialogue.append({"role": "assistant", "content": "\n\n".join(post["text"] for post in group)})
        else:
            dialogue.append({"role": "user", "content": json.dumps({"messages": group}, ensure_ascii=False)})

    for post in posts:
        is_bot = post["author"] == replace_speaker
        if group and is_bot != group_is_bot:
            add_group()
            group = []
        group_is_bot = is_bot
        if is_bot:
            group.append(post)
        else:
            group.append(post)
    add_group()
    if dialogue[-1]["role"] == "user":
        tail = json.loads(dialogue.pop()["content"])["messages"]
        print(f"Предупреждение: пропущено {len(tail)} сообщение(й) после последнего ответа "
              f"{replace_speaker!r}; у них нет ответной реплики бота.", file=sys.stderr)
    return [validate_dialogue(dialogue)]


def read_existing(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    data = yaml.safe_load(raw.decode("utf-8-sig"))
    if (not isinstance(data, dict) or not isinstance(data.get("speaker"), str)
            or not data["speaker"].strip() or not isinstance(data.get("dialogues"), list)):
        raise ValueError("Файл примеров должен содержать speaker и список dialogues")
    for dialogue in data["dialogues"]:
        validate_dialogue(dialogue)
    return raw, data


def append_dialogues(path: Path, incoming: list[list[dict[str, str]]], *, dry_run: bool) -> tuple[int, Path | None]:
    original, data = read_existing(path)
    additions: list[list[dict[str, str]]] = []
    for dialogue in incoming:
        validate_dialogue(dialogue)
        if dialogue not in data["dialogues"] and dialogue not in additions:
            additions.append(dialogue)
    if not additions:
        return 0, None

    line_ending = "\r\n" if b"\r\n" in original else "\n"
    fragment = yaml.safe_dump(additions, allow_unicode=True, sort_keys=False, width=1000)
    empty_header = re.search(rb"(?m)^dialogues:[ \t]*\[\][ \t]*(?:\r?\n|$)", original)
    if empty_header is not None:
        updated = (original[:empty_header.start()] + b"dialogues:" + line_ending.encode("ascii")
                   + fragment.replace("\n", line_ending).encode("utf-8")
                   + original[empty_header.end():])
    else:
        header = re.search(rb"(?m)^dialogues:[ \t]*\r?\n", original)
        if header is None:
            raise ValueError("Для сохранения старого форматирования требуется отдельная строка dialogues:")
        first_item = re.search(rb"(?m)^( *)-", original[header.end():])
        indentation = first_item[1].decode("ascii") if first_item is not None else ""
        inserted = textwrap.indent(fragment, indentation)
        updated = (original[:header.end()] + inserted.replace("\n", line_ending).encode("utf-8")
                   + original[header.end():])
    expected = dict(data, dialogues=additions + data["dialogues"])
    if yaml.safe_load(updated.decode("utf-8-sig")) != expected:
        raise ValueError("Новый YAML не прошёл проверку; исходный файл сохранён")

    if dry_run:
        print(fragment, end="")
        return len(additions), None

    # Never overwrite a file that changed while the input was being prepared.
    if path.read_bytes() != original:
        raise RuntimeError("Файл примеров изменился во время операции; повтори команду")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(f"{path.name}.{stamp}.bak")
    shutil.copy2(path, backup)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as temp:
            temp_path = Path(temp.name)
            temp.write(updated)
            temp.flush()
            os.fsync(temp.fileno())
        if path.read_bytes() != original:
            raise RuntimeError("Файл примеров изменился во время записи; повтори команду")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return len(additions), backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Добавить диалог в YAML с примерами Бямки")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES, help="Целевой YAML")
    parser.add_argument("--input", type=Path, help="JSON/YAML с dialogue или dialogues")
    parser.add_argument("--telegram-text", type=Path, help="Скопированный текст переписки Telegram")
    parser.add_argument("--replace-speaker", help="Имя участника, чьи реплики становятся ответами бота")
    parser.add_argument("--user", action="append", help="Реплика участника; можно повторять")
    parser.add_argument("--assistant", action="append", help="Ответ бота; можно повторять")
    parser.add_argument("--dry-run", action="store_true", help="Показать добавляемый YAML без записи")
    args = parser.parse_args()
    try:
        if args.telegram_text is not None:
            if args.input or args.user or args.assistant or not args.replace_speaker:
                parser.error("для --telegram-text укажи --replace-speaker; другие источники нельзя сочетать")
            incoming = read_telegram_text(args.telegram_text, args.replace_speaker)
        elif args.input is not None:
            if args.user or args.assistant or args.replace_speaker:
                parser.error("--input нельзя сочетать с --user/--assistant/--replace-speaker")
            incoming = read_input(args.input)
        else:
            if args.replace_speaker:
                parser.error("--replace-speaker используется только с --telegram-text")
            if not args.user or not args.assistant or len(args.user) != len(args.assistant):
                parser.error("укажи равное число --user и --assistant либо --input")
            dialogue = []
            for user, answer in zip(args.user, args.assistant):
                dialogue.extend([{"role": "user", "content": user},
                                 {"role": "assistant", "content": answer}])
            incoming = [validate_dialogue(dialogue)]
        added, backup = append_dialogues(args.examples, incoming, dry_run=args.dry_run)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")
    if added == 0:
        print("Диалоги уже есть в файле; изменений нет")
    elif backup is not None:
        print(f"Добавлено диалогов: {added}. Резервная копия: {backup}")


if __name__ == "__main__":
    main()
