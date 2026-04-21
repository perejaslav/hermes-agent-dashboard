# Hermes Agent Dashboard

Hermes Agent Dashboard — это небольшой самодостаточный дашборд для уже установленного Hermes Agent.
Он показывает аналитику сессий, использование токенов, распределение по моделям, инструменты, delegate-вызовы и другие live-метрики.

## Что есть в репозитории

- `hermes-dashboard.py` — сервер дашборда и встроенный UI
- `install.sh` — установка дашборда в существующий Hermes Agent
- `uninstall.sh` — удаление установленных файлов и systemd-сервиса
- `hermes-dashboard.service` — шаблон systemd unit, который использует установщик
- `README.md` — описание на английском

## Требования

Этот дашборд предназначен для человека, у которого уже установлен и работает Hermes Agent локально.
Ожидается Hermes-окружение вроде:

- `~/.hermes/state.db`
- `~/.hermes/sessions/`
- Python-окружение Hermes Agent на машине

Отдельный backend или база данных не нужны.
Дашборд читает локальное состояние Hermes напрямую.

## Быстрая установка

```bash
git clone https://github.com/perejaslav/hermes-agent-dashboard.git
cd hermes-agent-dashboard
sudo ./install.sh
```

Если Hermes установлен не в стандартном пути Python, можно указать его явно:

```bash
HERMES_PYTHON=/path/to/hermes-agent/venv/bin/python3 sudo ./install.sh
```

## После установки

- имя сервиса: `hermes-dashboard`
- порт: `8420`
- адрес: `http://localhost:8420`

Полезные команды:

```bash
sudo systemctl status hermes-dashboard
sudo systemctl restart hermes-dashboard
sudo journalctl -u hermes-dashboard -n 50
```

## Удаление

```bash
sudo ./uninstall.sh
```

Это удалит файл дашборда и systemd unit, но не тронет данные Hermes (`state.db`, sessions).

## Примечания

- Локализация работает на стороне клиента, отдельная i18n-библиотека не нужна.
- Дашборд — это один Python-файл со встроенными HTML/CSS/JS.
- Он использует данные Hermes только с локальной машины.
