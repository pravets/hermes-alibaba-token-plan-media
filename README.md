# hermes-alibaba-token-plan-media

Два провайдер-плагина для [Hermes Agent](https://hermes-agent.nousresearch.com/docs/), которые
подключают подписку **Alibaba Cloud Model Studio Token Plan** (`sk-sp-…`) к инструментам
`image_generate` и `video_generate`.

| Плагин | Провайдер | Модели |
|---|---|---|
| `image_gen/alibaba-token-plan` | `alibaba-token-plan` | `qwen-image-3.0-pro`, `wan2.7-image-pro`, `wan2.7-image` |
| `video_gen/alibaba-token-plan` | `alibaba-token-plan` | `wan3.0-video`, `wan3.0-video-prime`, `happyhorse-1.1-t2v`, `happyhorse-1.1-i2v`, `happyhorse-1.1-r2v` |

## Зачем отдельные плагины

Token Plan отдаёт картинки и видео **не** по OpenAI-совместимому протоколу: на
`/compatible-mode/v1/images/generations` и `/videos/generations` приходит 404. Генерация живёт
на нативных DashScope-путях того же хоста:

- картинки — синхронно `POST /api/v1/services/aigc/multimodal-generation/generation`;
- видео — асинхронно `POST /api/v1/services/aigc/video-generation/video-synthesis`
  (`X-DashScope-Async: enable`) и опрос `GET /api/v1/tasks/{task_id}`.

Плагины инкапсулируют это в штатные ABC Hermes: `agent.image_gen_provider.ImageGenProvider` и
`agent.video_gen_provider.VideoGenProvider`. Агент видит один инструмент, бэкенд выбирается
конфигом.

Документация Alibaba:

- [Интеграция Token Plan](https://help.aliyun.com/en/model-studio/token-plan-integration-overview)
- [Картинки в Token Plan](https://help.aliyun.com/en/model-studio/token-plan-image-generation)
- [Видео в Token Plan](https://help.aliyun.com/en/model-studio/token-plan-video-generation)
- [Wan 3.0 Video API](https://help.aliyun.com/en/model-studio/wan3-video-generation-api-reference)

## Установка

```bash
git clone https://github.com/pravets/hermes-alibaba-token-plan-media.git
cd hermes-alibaba-token-plan-media
bash scripts/install.sh            # симлинки в ~/.hermes/plugins/{image_gen,video_gen}/

hermes plugins enable image_gen/alibaba-token-plan
hermes plugins enable video_gen/alibaba-token-plan
hermes tools                       # Image Generation / Video Generation → Alibaba Token Plan
```

`hermes tools` показывает оба плагина отдельными строками, спрашивает ключ и записывает модель в
`image_gen.model` / `video_gen.model`.

## Ключ

Ключ медиа-генерации **независим** от ключей чат-провайдеров: плагины не читают
`HERMES_CUSTOM_*`, `DASHSCOPE_API_KEY` и прочие чужие секреты. Источники по приоритету:

1. `ALIBABA_TOKEN_PLAN_MEDIA_API_KEY` — переменная окружения (её и спрашивает `hermes tools`,
   значение попадает в `$HERMES_HOME/.env`);
2. файл `~/.alibaba/token-plan-media.key` (первая непустая строка, `chmod 600`), путь
   переопределяется `ALIBABA_TOKEN_PLAN_MEDIA_KEY_FILE`.

```bash
install -d -m 700 ~/.alibaba
printf '%s\n' 'sk-sp-…' > ~/.alibaba/token-plan-media.key
chmod 600 ~/.alibaba/token-plan-media.key
```

Ключ `sk-sp-` привязан к региону: Personal Edition — только Singapore
(`https://token-plan.ap-southeast-1.maas.aliyuncs.com`). Cross-region вызов даёт 401.

## Настройка

Все ключи читаются из `config.yaml`; менять через `hermes config set`.

### Картинки — `image_gen.alibaba-token-plan.*`

| Ключ | По умолчанию | Что делает |
|---|---|---|
| `model` | `qwen-image-3.0-pro` | модель (`wan2.7-image-pro`, `wan2.7-image`) |
| `size` | из aspect | `W*H` или пресет `1K`/`2K`/`4K`; по умолчанию landscape `2048*1152`, square `2048*2048`, portrait `1152*2048` |
| `n` | `1` | сколько картинок (1–4, API ограничивает) |
| `watermark` | `false` | водяной знак вендора |
| `prompt_extend` | `true` | LLM-переписывание промпта |
| `base_url` | Singapore-хост | workspace-домен вида `https://<id>.ap-southeast-1.maas.aliyuncs.com` |
| `request_timeout` | `180` | таймаут запроса, сек |

Переопределения окружением: `ALIBABA_IMAGE_MODEL`, `ALIBABA_TOKEN_PLAN_BASE_URL`.

### Видео — `video_gen.alibaba-token-plan.*`

| Ключ | По умолчанию | Что делает |
|---|---|---|
| `model` | `wan3.0-video` | семейство (см. таблицу моделей) |
| `resolution` | из инструмента | `480P`/`720P`/`1080P` (алиасы `480p`, `1080p`, …) |
| `poll_interval` | `8` | период опроса задачи, сек (минимум 2) |
| `poll_timeout` | `900` | общий бюджет ожидания, сек (минимум 60) |
| `download` | `true` | скачивать mp4 в `$HERMES_HOME/cache/generated/videos` (ссылка живёт 24 ч) |
| `watermark` / `prompt_extend` | `false` / `true` | параметры API |
| `submit_timeout` | `120` | таймаут постановки задачи, сек |

Переопределение модели: `ALIBABA_VIDEO_MODEL`.

## Возможности моделей

| Семейство | Вход | Длительность | Разрешения | Особенности |
|---|---|---|---|---|
| `wan3.0-video` | текст, первый кадр, первый+последний, референсы | 2–30 с (`duration=0` → smart) | 480P/720P/1080P | native audio, `ratio` до 21:9 + `adaptive` |
| `wan3.0-video-prime` | то же | 2–30 с | 480P/720P/1080P | ускоренная версия |
| `happyhorse-1.1-t2v` | текст | 3–15 с | 480P/720P/1080P | кинематографичный t2v |
| `happyhorse-1.1-i2v` | ровно один первый кадр | 3–15 с | 480P/720P/1080P | `ratio` не передаётся — следует за картинкой |
| `happyhorse-1.1-r2v` | референс-картинки | 3–15 с | 480P/720P/1080P | консистентность персонажа |

`negative_prompt` у видео-эндпоинта Token Plan нет — провайдер его не отправляет и сообщает об
этом через `capabilities()` (`supports_negative_prompt: False`).

Источники изображений для правки/анимации принимаются как публичный URL или локальный файл
(конвертируется в `data:` URI, лимит 20 МБ).

## Проверка

```bash
uv venv .venv --python /usr/bin/python3
uv pip install --python .venv/bin/python pytest requests
.venv/bin/python -m pytest -q
```

Тесты работают на мок-транспорте: проверяют тела запросов, роутинг t2v/i2v, маппинг
aspect → size, клампы длительности и разрешения, состояния задач и чистые `error_response`
вместо трейбеков. Реальных платных вызовов нет.

Живая проверка ключа (списывает кредиты плана, запускать интерпретатором Hermes):

```bash
python scripts/smoke.py --image     # одна картинка, ~30 с
python scripts/smoke.py --video     # ролик 3 с 480P, ~2 мин
```

Реальный прогон `wan3.0-video` (480P, 3 с) на этой реализации: задача отработала за 109 с,
на выходе MP4 h264+aac 854×480, 3.02 с, файл скачан в `$HERMES_HOME/cache/generated/videos/`.

## Структура

```
hermes_tpm/          общий слой: ключ, base_url, HTTP, media-URL, конфиг
image_gen/           провайдер картинок + plugin.yaml
video_gen/           провайдер видео + plugin.yaml
scripts/             install.sh / uninstall.sh / smoke.py
tests/               pytest (agent.* подменяется заглушками вне Hermes)
```

## Ограничения

- План Token Plan предназначен для интерактивной работы агент-инструментов; массовые
  неинтерактивные прогоны нарушают условия подписки.
- Ссылки на результаты живут 24 часа — плагины по умолчанию скачивают артефакты локально.

## Лицензия

MIT, см. [LICENSE](LICENSE).
