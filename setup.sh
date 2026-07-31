#!/usr/bin/env bash
# Разворачивание на выданной машине. По регламенту 1.4 личные компьютеры на площадке
# запрещены, значит всё поднимается с нуля одной командой на чужом железе.
#
#   ./setup.sh            проверить окружение и прогнать тесты
#   ./setup.sh --deps     ещё и доставить недостающие пакеты
#
# Бортовое ПО дрона намеренно обходится стандартной библиотекой: ни flask, ни requests,
# ни PyYAML для полёта не нужны (docs/organizer_handouts.md §2.3 — установка на борт
# требует смены режима Wi-Fi и теряет доступ к 172.17.49.2).
set -u

КОРЕНЬ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$КОРЕНЬ"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

echo "== Окружение =="
"$PY" --version || { echo "Python не найден. Задайте PYTHON=/путь/к/python"; exit 1; }
echo "Репозиторий: $КОРЕНЬ"

echo
echo "== Библиотеки =="
проверить() {
    if "$PY" -c "import $1" >/dev/null 2>&1; then
        echo "  ЕСТЬ  $1   ($2)"
        return 0
    fi
    echo "  НЕТ   $1   ($2)"
    return 1
}

нужно_ставить=()
проверить numpy  "нужен перцепции и заглушке камеры"        || нужно_ставить+=("numpy")
проверить cv2    "нужен записи попыток и перцепции"          || нужно_ставить+=("opencv-python")
проверить pytest "нужен только для прогона тестов"           || нужно_ставить+=("pytest")
проверить flask  "необязателен: хаб умеет работать на stdlib" || true
проверить yaml   "необязателен: config.py читает YAML сам"    || true

if [ "${1:-}" = "--deps" ] && [ ${#нужно_ставить[@]} -gt 0 ]; then
    echo
    echo "== Доставляю: ${нужно_ставить[*]} =="
    # Без venv: на борту в виртуальном окружении pioneer_sdk2 не работает, и одна
    # и та же команда должна годиться и там, и на выданной машине.
    "$PY" -m pip install --user "${нужно_ставить[@]}"
fi

echo
echo "== Тесты =="
if "$PY" -c "import pytest" >/dev/null 2>&1; then
    "$PY" -m pytest tests -q || { echo "ТЕСТЫ УПАЛИ — разбираться до вылета"; exit 1; }
else
    echo "  pytest не установлен, пропускаю (./setup.sh --deps поставит)"
fi

echo
echo "== Сквозная проверка без дрона =="
echo "  1) python3 'распределительный хаб/hub.py'"
echo "  2) PIONEER_MOCK=1 python3 квадрокоптер/main.py --hub http://127.0.0.1:5001"
echo "  3) python3 'распределительный хаб/transmitter.py' --hub http://127.0.0.1:5001 takeoff"
echo
echo "== На борту, день 1 =="
echo "  scp tools/check_sdk.py pioneermini@172.17.49.2:~/workspace/"
echo "  ssh pioneermini@172.17.49.2 'cd workspace && python3 check_sdk.py'"
echo "  результаты -> квадрокоптер/config.yaml (метки «НЕ ПРОВЕРЕНО»)"
echo
echo "Готово."
