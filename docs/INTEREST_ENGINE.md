# Движок интересов Бямки

## Что добавлено

В прототип добавлен отдельный слой `personality_engine`, который даёт боту не только память, но и внутренние склонности к темам.

Новые файлы:

```text
bot_app/personality/interest_field.py     # seeded Perlin-like поле интереса по embedding
bot_app/personality/interest_engine.py    # оценка интереса + эволюция learned interests
bot_app/personality/fingerprint.py        # 2D SVG fingerprint по seed характера
scripts/generate_fingerprint.py           # CLI-генерация fingerprint
```

Новые команды Telegram:

```text
/fingerprint  # отправляет SVG-картинку fingerprint характера
/interests    # показывает самые выраженные learned interests текущего чата
```

## Общая идея

У каждого сообщения строится embedding через Ollama. Затем embedding проецируется в маленькое латентное пространство и прогоняется через несколько seeded Perlin-like шумовых полей.

Получается профиль:

```text
innate_interest  # врождённый интерес по seed
learned_interest # приобретённый интерес по похожим прошлым темам
final_interest   # итоговая смесь врождённого, приобретённого и новизны
curiosity         # насколько хочется развить тему
humor             # насколько уместна короткая шутка
memory            # насколько стоит внимательнее сохранять знания
reply             # насколько тема подталкивает к ответу
novelty           # насколько тема не похожа на уже знакомые интересы
```

Важно: интерес не является приказом отвечать. Это bias характера. Router всё равно учитывает уместность, прямое обращение, cooldown и антиспам.

## Почему не обычный random

Обычный random сделал бы поведение рваным: похожие темы могли бы получать совершенно разные оценки.

Perlin-like поле гладкое: близкие embedding-точки получают похожие значения. Поэтому у Бямки появляются устойчивые зоны интереса: если ей «резонируют» локальные LLM, память и агентная архитектура, то соседние темы тоже будут чаще вызывать интерес.

## Эволюция интересов

Кроме врождённого шума есть learned interests.

В SQLite добавлены таблицы:

```sql
interest_observations
interest_topics
```

`interest_observations` хранит оценки отдельных сообщений.

`interest_topics` хранит темы, которые Бямка начала считать интересными в конкретном чате:

```text
topic_text
embedding_json
learned_interest
curiosity
humor
memory
reply
seen_count
created_at
last_seen_at
```

Если новая тема похожа на уже существующую, `learned_interest` растёт плавно. Если тема новая, создаётся новая запись.

## Влияние на router

Первый router теперь видит:

```text
interest_profile=innate=0.71, learned=0.20, final=0.56, curiosity=0.80, humor=0.33, memory=0.72, reply=0.66, novelty=0.90
```

Это влияет так:

- высокий `reply` может немного снизить порог spontaneous reply;
- высокий `humor` может разрешить короткий light_humor;
- высокий `memory` повышает шанс подтянуть память и чуть снижает порог авто-запоминания;
- низкий интерес не запрещает ответ, если бота прямо спросили.

## Второй router после памяти

Теперь после первого решения `reply/tool_reply/ask_clarify` бот подтягивает:

- behavior facts;
- topic facts;
- interest profile.

После этого вызывается `router.review_after_memory(...)`.

Этот второй router может вернуть `ignore`, если после памяти видно, что:

- факты нерелевантны;
- ответ будет навязчивым;
- сообщение короткое и не требует реакции;
- бот будет повторять уже известные факты.

По умолчанию второй router не может игнорировать прямое обращение в личке/reply/mention:

```yaml
routing:
  allow_post_memory_ignore_direct: false
```

## Fingerprint

Fingerprint — это 2D SVG-карта seeded noise-полей. Она не влияет на диалог напрямую, но визуализирует «характер» seed-а.

CLI:

```bash
python scripts/generate_fingerprint.py --config config/config.example.yaml --output data/interest_fingerprint.svg
```

Telegram:

```text
/fingerprint
```

При смене `personality_engine.seed` fingerprint меняется.

## Конфиг

Основной блок:

```yaml
personality_engine:
  enabled: true
  seed: 73421
  latent_dim: 6
  noise_scale: 2.7
  octaves: 4
  persistence: 0.55

  learned_similarity_threshold: 0.72
  learned_topics_limit: 5
  topic_merge_similarity: 0.86
  learn_rate: 0.08
  learn_decay: 0.015

  boost_reply_threshold: 0.66
  spontaneous_reply_threshold: 0.84
  memory_interest_threshold: 0.76
  save_interest_threshold: 0.72
  router_threshold_discount: 0.12

  fingerprint_size: 96
  fingerprint_cell_px: 5
```

## Практические замечания

1. `latent_dim` лучше держать в диапазоне 4–8. Чем больше размерность, тем дороже n-dimensional noise.
2. `seed` задаёт врождённый характер. Менять seed — всё равно что создать другого Бямку.
3. Если бот слишком часто влезает, подними `spontaneous_reply_min_confidence`, `boost_reply_threshold` или `spontaneous_reply_threshold`.
4. Если бот плохо развивает интересы, снизь `save_interest_threshold` или `topic_merge_similarity`.
5. Learned interests привязаны к `chat_id`, чтобы не смешивать характер поведения между разными чатами.

## Анти-зацикливание в новой версии

В этой версии интересы защищены от липких тематических аттракторов.

Добавлены:

- `topic_cooldown_minutes` и `repeated_topic_penalty` — если похожая тема была недавно, она получает штраф;
- `max_innate_contribution` — врождённый шум не может сам по себе слишком сильно поднять итоговый интерес;
- `novelty_seen_count_strength` и `max_novelty_penalty` — часто повторяющиеся темы становятся менее новыми;
- `learn_negative_rate` и `learn_repetition_decay` — learned-interest теперь может затухать;
- `repetition_bias_block_threshold` — при высоком repetition penalty router не получает interest-бонус.

Новые поля в `InterestProfile`:

```text
novelty_penalty
repetition_penalty
```

Главное правило теперь такое:

```text
интерес к теме влияет на внимание, но не даёт права постоянно возвращать эту тему в ответы
```

Подробнее: `docs/ANTI_LOOP_CHANGES.md`.
