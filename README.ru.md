# Hermes Agent Dashboard

Небольшой самодостаточный аналитический дашборд для [Hermes Agent](https://github.com/nousresearch/hermes-agent). Визуализирует сессии, токены, производительность моделей, вызовы инструментов, активность субагентов и временные паттерны — всё из локальной установки Hermes.

## Возможности

| Вкладка | Что показывает |
|---------|---------------|
| **Overview** | KPI-карточки, распределение моделей, топ инструментов, активность по дням, тренды токенов, scorecard моделей |
| **Sessions** | Полная таблица сессий с поиском, сортировкой, pinning'ом, топ по длительности и токенам |
| **Tools** | Частота инструментов, heatmap корреляции, почасовая активность, топ сессий по tool-вызовам |
| **Subagents** | Дерево delegate'ов, распределение целей, toolsets субагентов, почасовая активность |
| **Trends** | Сравнение периодов (текущий vs предыдущий), delta-бейджи, overlaid графики |
| **Analytics** | Активность по часам, donut по платформам, stacked bar platform × hour |

### Встроенные функции

- **Real-time** — WebSocket push или polling каждые 30 секунд
- **Глобальный поиск** — фильтрация по модели, платформе, инструменту, имени файла
- **Pin сессий** — закрепление важных сессий (сохраняется в localStorage)
- **Drill-down** — клик по сессии открывает полный timeline с ролями, tool calls, токенами
- **Alert banner** — anomaly detection (>2σ) подсвечивает аномальные метрики
- **CSV export** — скачивание таблиц sessions, duration, tokens
- **Hotkeys** — `R` обновить, `/` поиск, `1–5` период
- **i18n** — русский / английский интерфейс

## Требования

- Установленный Hermes Agent
- `~/.hermes/state.db` (SQLite)
- `~/.hermes/sessions/*.jsonl`
- Python 3.11+ с FastAPI + uvicorn

Отдельный backend или база данных не нужны.

## Быстрая установка

```bash
git clone https://github.com/perejaslav/hermes-agent-dashboard.git
cd hermes-agent-dashboard
sudo ./install.sh
```

Если Hermes установлен не в стандартном пути:

```bash
HERMES_PYTHON=/path/to/hermes-agent/venv/bin/python3 sudo ./install.sh
```

## После установки

- **Сервис:** `hermes-dashboard`
- **Порт:** `8420`
- **Адрес:** `http://localhost:8420`

```bash
sudo systemctl status hermes-dashboard
sudo systemctl restart hermes-dashboard
sudo journalctl -u hermes-dashboard -n 50
```

## Удаление

```bash
sudo ./uninstall.sh
```

Удаляет файл дашборда и systemd unit. Данные Hermes (`state.db`, sessions) не трогает.

## Архитектура

- **Один файл:** `hermes-dashboard.py` (~2600 строк) — FastAPI backend + inline HTML/CSS/JS
- **Фронтенд:** vanilla JS + Chart.js 4.x (без build-шага)
- **Источники данных:** SQLite `state.db` + JSONL файлы сессий
- **Деплой:** systemd service с auto-restart

## Лицензия

MIT
