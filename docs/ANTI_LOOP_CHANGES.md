# Анти-зацикливание интересов

Эта итерация дорабатывает `personality_engine`, чтобы внутренние интересы Бямки не превращались в навязчивые темы.

Реализованы пункты:

1. **Topic cooldown** — похожая тема, которую бот недавно уже видел, получает `repetition_penalty`.
2. **Ограничение innate interest** — врождённый шум больше не может сам по себе сильно поднять итоговый интерес.
3. **Novelty penalty** — темы, которые много раз встречались в чате, становятся менее новыми и слабее подталкивают бота к ответу.
4. **Двусторонняя эволюция learned interest** — learned-interest теперь может не только расти, но и затухать при повторении, низкой новизне и cooldown penalty.

## Почему это нужно

Seeded noise создаёт устойчивые зоны интереса. Это хорошо для характера, но опасно для поведения: если тема попала в яркую область поля, бот может начать возвращаться к ней слишком часто.

Поэтому теперь действует правило:

```text
интерес к теме ≠ разрешение постоянно её упоминать
```

Интерес влияет на внимание, вероятность ответа и запоминание. Но повтор темы снижает её новизну и блокирует бонусы роутера.

## Новые поля профиля интереса

`InterestProfile` теперь содержит дополнительные сигналы:

```text
novelty_penalty     # штраф за то, что похожая тема уже часто встречалась
repetition_penalty  # штраф за то, что похожая тема была недавно
```

Пример лога:

```text
innate=0.82, learned=0.61, final=0.39, curiosity=0.70, humor=0.44,
memory=0.55, reply=0.76, novelty=0.02, novelty_penalty=0.31, repetition_penalty=0.26
```

Это означает: зона боту может быть врождённо интересна, но тема уже заезжена или недавно была, поэтому итоговый `final_interest` снижен.

## Topic cooldown

Если похожая тема уже есть в `interest_topics` и её `last_seen_at` находится внутри окна cooldown, применяется штраф:

```yaml
personality_engine:
  topic_cooldown_minutes: 180
  repeated_topic_penalty: 0.35
```

Штраф максимален сразу после повторения темы и плавно затухает к концу окна.

## Ограничение влияния врождённого интереса

Врождённый шум теперь ограничен через `max_innate_contribution`:

```yaml
personality_engine:
  innate_weight: 0.20
  max_innate_contribution: 0.12
```

Даже если шум считает тему очень интересной, он может дать только небольшой плюс. Это сохраняет характер, но не даёт seed-у управлять всем поведением.

## Novelty penalty

Чем чаще похожая тема встречалась, тем меньше её новизна:

```yaml
personality_engine:
  novelty_seen_count_strength: 0.08
  max_novelty_penalty: 0.45
```

Это защищает от ситуации, когда бот постоянно возвращает в диалог одну и ту же любимую область.

## Двусторонняя эволюция learned interest

Раньше похожая тема почти всегда усиливала `learned_interest`. Теперь оценка может падать:

```yaml
personality_engine:
  learn_rate: 0.08
  learn_decay: 0.015
  learn_negative_rate: 0.04
  learn_repetition_decay: 0.10
  min_learned_interest: 0.02
```

Упрощённо:

```text
если final_interest высокий и тема не зациклена:
    learned_interest немного растёт

если final_interest низкий или repetition_penalty высокий:
    learned_interest падает
```

## Влияние на router

Если `repetition_penalty` выше порога, interest-бонусы роутера отключаются:

```yaml
personality_engine:
  repetition_bias_block_threshold: 0.18
```

То есть высокий `reply` или `humor` больше не заставит бота спонтанно отвечать, если похожая тема уже недавно всплывала.

## Prompt-правило для responder

В `ContextBuilder._system_prompt()` добавлено правило:

```text
Внутренние интересы — это фон внимания, а не список любимых тем для ответа.
Не возвращайся к одной и той же теме или образу несколько раз подряд.
Если repetition_penalty или novelty_penalty высокие, не возвращайся к похожей теме и отвечай суше.
```

Это снижает шанс, что модель будет тащить любимую тему как метафору в каждый ответ.

## Что настраивать, если бот всё ещё зацикливается

Усиль анти-зацикливание:

```yaml
personality_engine:
  topic_cooldown_minutes: 360
  repeated_topic_penalty: 0.45
  novelty_seen_count_strength: 0.12
  learn_repetition_decay: 0.16
  repetition_bias_block_threshold: 0.12
```

Ослабь врождённый шум:

```yaml
personality_engine:
  innate_weight: 0.12
  max_innate_contribution: 0.07
```

Сделай спонтанные ответы реже:

```yaml
routing:
  spontaneous_reply_min_confidence: 0.82

personality_engine:
  spontaneous_reply_threshold: 0.90
```
