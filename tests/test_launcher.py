# -*- coding: utf-8 -*-
"""Проверки пульта (launch.py).

Сам запуск процессов тестами не проверить — нужен борт. Зато проверяется то,
что на площадке ломается молча: состав заливки (забытый файл обнаружится не здесь,
а в воздухе) и ключи ssh, от которых зависит, сядет ли дрон по Ctrl+C.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import launch  # noqa: E402


def имена(план):
    return {os.path.basename(f) for файлы, _ in план for f in файлы}


def test_заливка_везёт_всё_нужное_для_полёта():
    план = launch.план_заливки()
    должны_быть = {"main.py", "mission.py", "flight.py", "survey.py", "config.py",
                   "mock_pioneer.py", "config.yaml", "tasks.py",
                   # Пункт 2: без любого из них дрон взлетит, но станций не найдёт
                   "localization.py", "fusion.py", "calib.py", "detector.py",
                   "pipeline.py",
                   "messages.py", "transport.py", "__init__.py",
                   "hub.py", "transmitter.py"}
    assert должны_быть <= имена(план)


def test_perception_кладётся_в_свою_папку():
    """perception/ — пакет: плоско рядом с main.py импорт `from perception.calib
    import ...` не соберётся, и дрон улетит без детекции."""
    план = launch.план_заливки()
    куда = {}
    for файлы, подпапка in план:
        for f in файлы:
            куда.setdefault(os.path.basename(f), []).append(подпапка)
    assert куда["calib.py"] == ["perception"]
    assert куда["detector.py"] == ["perception"]
    assert куда["pipeline.py"] == ["perception"]
    # __init__.py нужен обоим пакетам, и оба обязаны уехать
    assert sorted(куда["__init__.py"]) == ["perception", "protocol"]


def test_protocol_кладётся_в_свою_папку():
    план = launch.план_заливки()
    куда = {os.path.basename(f): подпапка for файлы, подпапка in план for f in файлы}
    # protocol/ — единый источник правды для дрона, передатчика и визуализатора;
    # плоско рядом с main.py он бы перекрыл импорт `from protocol.messages import ...`
    assert куда["messages.py"] == "protocol"
    assert куда["transport.py"] == "protocol"
    assert куда["check_sdk.py"] == "tools"
    assert куда["main.py"] == ""


def test_keep_config_не_трогает_конфиг_на_борту():
    # На площадке config.yaml правится по месту (габариты трассы рулеткой),
    # и перезалить его поверх — значит молча вернуть старые размеры.
    assert "config.yaml" in имена(launch.план_заливки(конфиг=True))
    assert "config.yaml" not in имена(launch.план_заливки(конфиг=False))


def test_все_файлы_плана_существуют():
    for файлы, _ in launch.план_заливки():
        assert файлы, "пустая группа: файл переименован или удалён"
        for f in файлы:
            assert os.path.exists(f)


def test_ssh_форсирует_tty_для_бортовых_процессов():
    # Без -tt удалённый python переживает обрыв ssh и остаётся висеть, держа камеру;
    # с -tt в него уходит Ctrl+C и дрон садится штатно.
    assert "-tt" in launch.ssh("python3 main.py", tty=True)
    assert "-tt" not in launch.ssh("true")


def test_ssh_batch_не_спрашивает_пароль():
    argv = launch.ssh("true", batch=True)
    assert "BatchMode=yes" in argv
    assert argv[-1] == "true"
    assert argv[-2] == "pioneermini@172.17.49.2"


def test_ssh_принимает_чужой_борт():
    argv = launch.ssh("true", хост="10.42.0.1", пользователь="pi")
    assert "pi@10.42.0.1" in argv


def test_обязательные_пакеты_совпадают_с_requirements():
    # launch.py --setup заменил setup.sh и остался единственным местом, которое
    # разворачивает выданную машину: пусть ставит ровно то, что объявлено.
    путь = os.path.join(ROOT, "requirements.txt")
    with open(путь, encoding="utf-8") as f:
        объявлено = {строка.strip() for строка in f
                     if строка.strip() and not строка.startswith("#")}
    обязательные = {пакет for _, пакет, _, нужен in launch.ПАКЕТЫ if нужен}
    assert обязательные == объявлено


def test_команда_на_взлёт_это_пакет_протокола():
    # Пульт не имеет права знать «свою» версию команды: в финале вместо передатчика
    # встаёт автомобиль, и дрон обязан принять от него ровно тот же пакет.
    from protocol.messages import parse

    пакет = launch.пакет_взлёта()
    разобрано = parse(пакет)
    assert разобрано["type"] == "takeoff"
    assert разобрано["src"] == "transmitter"


@pytest.mark.parametrize("аргументы", [
    [],
    ["--mock"],
    ["--hub", "local", "--takeoff"],
    ["--push", "-y"],
])
def test_разбор_аргументов(аргументы, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launch.py"] + аргументы)
    parser = None
    # main() запускать нельзя (он полезет на борт) — проверяем только парсер,
    # собранный тем же кодом.
    import argparse
    исходный = argparse.ArgumentParser.parse_args

    def перехват(self, args=None, namespace=None):
        nonlocal parser
        parser = self
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", перехват)
    with pytest.raises(SystemExit):
        launch.main()
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", исходный)
    разобрано = parser.parse_args(аргументы)
    assert разобрано.port == 5001
    assert разобрано.drone == "172.17.49.2"
