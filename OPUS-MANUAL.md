# Opus — короткая шпаргалка

Один файл на всё. Если что-то непонятно — открывай этот.

## Где что лежит

| Что | Путь |
|---|---|
| Код | `D:\opus\backend\`, `D:\opus\frontend\` |
| Конфиг | `D:\opus\.env` |
| Питон-окружение | `D:\opus\.venv\` |
| Сырые снапшоты (24/7 коллектор) | `D:\opus-data\snapshots\<SYMBOL>\<YYYY-MM-DD>\<HH>.parquet` |
| Обученные модели | `D:\opus\models\global\h<HORIZON>\` (`model.lgb`, `meta.json`, `eval.json`) |
| Лог paper-сделок | `D:\opus-data\trades\<YYYY-MM-DD>.parquet` (UTC) |
| Логи бота | `D:\opus\logs\` |
| Web UI | `http://127.0.0.1:8080` |

## Главный launcher — `opus.bat`

Лежит в `D:\opus\scripts\opus.bat`. Запускай так (можно из любой папки):

```
D:\opus\scripts\opus.bat <команда> [аргументы]
```

Или добавь `D:\opus\scripts` в `PATH` Windows и пиши просто `opus <команда>`.

### Все команды

| Команда | Что делает |
|---|---|
| `opus help` | Показать список команд |
| `opus start` | Запустить бот в текущем окне (Ctrl+C — остановить) |
| `opus stop` | Найти и убить запущенный бот |
| `opus restart` | stop + start |
| `opus status` | Показать: запущен/нет + текущий конфиг (mode, символы, threshold) |
| `opus logs` | Хвост последнего лога с авто-обновлением |
| `opus open-ui` | Открыть `http://127.0.0.1:8080` в браузере |
| `opus trades` | Сводка по сегодняшним paper-сделкам |
| `opus trades 2026-04-28` | Сводка за конкретную дату |
| `opus config` | Показать **весь** конфиг в JSON |
| `opus train [args]` | Переобучить модели (см. ниже) |
| `opus backtest [args]` | Прогнать бэктест на test-слайсе (см. ниже) |

### `opus train` — обучение

По умолчанию учит **3 горизонта (1s/5s/30s)** на **всех** монетах из `D:\opus-data\snapshots\`:

```
opus train
```

С кастомными горизонтами (любые числа, `ms` / `s` / `m`):

```
opus train --horizons 2s 5s 15s
opus train --horizons 500ms 1s 3s
opus train --horizons 1m 5m
```

Только некоторые монеты:

```
opus train --symbols AIOTUSDT BTCUSDT NEIROUSDT
```

Только за свежий период (последние 7 дней):

```
opus train --from-date 2026-04-22
```

Per-symbol модели вместо одной глобальной:

```
opus train --per-symbol --symbols AIOTUSDT
```

С Optuna-поиском гиперпараметров (медленнее, но качественнее):

```
opus train --optuna-trials 16
```

Модели падают в `D:\opus\models\global\h<HORIZON>\`. После обучения **перезапусти бот** (`opus restart`).

### `opus backtest` — оценка модели на test-слайсе

```
opus backtest                                       # автоопределение горизонтов из папки моделей
opus backtest --horizons 2s 15s                     # только эти горизонты
opus backtest --symbols AIOTUSDT NEIROUSDT          # только эти монеты
opus backtest --target-trade-frac 0.05              # порог выбираем под 5% строк (~частота сделок)
```

Печатает таблицу `taker / maker_best / maker_zero / maker_cross` по каждому горизонту. JSON-отчёты падают в `D:\opus\models\global\backtest\`.

## Конфиг — что значит каждая строка `.env`

Полная справка — `.env.example`. Главное:

```ini
# режим работы
OPUS_MODE=collect    # или paper / live
                     # collect = только пишем снапшоты (без сделок)
                     # paper   = ML предсказания + paper-сделки в parquet
                     # live    = реальные ордера на Binance Futures (НЕ ВКЛЮЧАЙ ПОКА)

# где данные / модели
OPUS_DATA_DIR=D:/opus-data
OPUS_MODEL_DIR=./models/global

# что торгуем (только в paper/live режиме)
OPUS_TRADE_HORIZON=1s              # на каком горизонте: 1s / 5s / 30s или любой который ты обучил
OPUS_TRADE_SYMBOLS=AIOTUSDT        # CSV: какие монеты разрешено торговать.
                                   # Пустой список = все подписанные.
                                   # ВАЖНО: эти монеты ДОЛЖНЫ быть подписаны через UI.
OPUS_TRADE_CONF_THRESHOLD=0.47     # минимальный |confidence| модели для входа
OPUS_TRADE_NOTIONAL_USD=1          # размер позиции в USD (для paper - чисто учётный)

# комиссии (для paper PnL и backtest)
OPUS_TAKER_FEE_BP=4                # 4 bp/side у Binance Futures обычного аккаунта
```

Список **подписанных** монет (откуда идут снапшоты) — задаётся **в UI** в панели `Symbols`. Это **не то же самое** что `OPUS_TRADE_SYMBOLS`. Бот всегда подписан на всё что в UI; `OPUS_TRADE_SYMBOLS` — фильтр для трейдера.

## Типичные сценарии

### "Хочу начать с нуля и собрать данные"

```
1. установи (один раз):  scripts\install_windows.ps1
2. отредактируй .env: OPUS_MODE=collect
3. opus start
4. открой UI:            opus open-ui
5. в UI добавь монеты которые хочешь собирать
6. ждёшь 1-3 дня
```

### "Хочу обучить модель на собранных данных"

```
opus stop                           # бот может писать дальше пока учим, но это сожрёт CPU
opus train                          # ~5-15 мин на 1.5М строк
opus start
```

### "Хочу обучить нестандартные горизонты — 2с, 7с, 1м"

```
opus train --horizons 2s 7s 1m
```

### "Хочу запустить paper-trading"

```
opus stop
notepad .env       # выставь OPUS_MODE=paper, OPUS_TRADE_SYMBOLS=AIOTUSDT, OPUS_TRADE_HORIZON=1s
opus start
```

Через 1-2 часа:
```
opus trades                         # сводка
```

### "Хочу проверить как модель работает на новой монете которой не было в обучении"

**Важно**: дефолтная модель использует `_symbol_id` как фичу — для unseen монет это даёт мусорные предсказания. Если хочешь чтобы модель **переносилась** на новые монеты, переобучи без этого признака:

```
opus train --no-symbol-feature --models-dir models\agnostic
```

Затем в `.env` поставь `OPUS_MODEL_DIR=./models/agnostic` и перезапусти бота. Эта модель будет работать на любой подписанной монете без переобучения, но AUC чуть ниже (~1-3 пункта).

Если просто хочешь быстро проверить дефолтную модель на новой монете:
```
1. в UI добавь новую монету (через панель Symbols)
2. подожди 30 мин (накопится 600+ снапшотов = predictor warmup готов)
3. opus stop
4. в .env:  OPUS_TRADE_SYMBOLS=NEWCOINUSDT
5. opus start
6. через 2-3 часа:  opus trades
```
Будет видно насколько просядет качество — обычно сильно.

### "Хочу проверить какие настройки сейчас активны"

```
opus status         # короткое summary
opus config         # всё в JSON
```

## Troubleshoot — типовые проблемы

| Что вижу | Что делать |
|---|---|
| `ERROR: venv not found` | `scripts\install_windows.ps1` (один раз) |
| `ERROR: .env not found` | скопируй `.env.example` -> `.env`, отредактируй |
| `JSONDecodeError` на старте | старая версия `config.py`. Обнови до `opus-toolkit-v2`. |
| `predictor: cold-start, skipping` в логах | predictor ждёт 600 снапшотов на символ. ~2.5 мин после старта |
| `0 paper trades за час` | подними порог входа: `OPUS_TRADE_CONF_THRESHOLD=0.30` или ниже |
| `RuntimeError: lightgbm not installed` | `.venv\Scripts\pip install lightgbm scikit-learn optuna` |
| `RuntimeError: no parquet files found` | сначала собери данные (`OPUS_MODE=collect`) хотя бы 1 день |
| Бот не останавливается через `opus stop` | `taskkill /F /IM python.exe` (агрессивно) |
| UI не открывается на `127.0.0.1:8080` | `opus status` — бот должен быть RUNNING; проверь `logs\*.log` |

## Дальнейшее чтение

- `README.md` — общая архитектура и история проекта
- `.env.example` — полный список **всех** настроек с комментариями
- `AGENTS.md` — заметки для AI-ассистентов о репо (можно игнорировать)
