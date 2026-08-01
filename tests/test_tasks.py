# -*- coding: utf-8 -*-
"""Профили подзадач: что включено в каждой попытке.

Главное, что здесь проверяется, — что подзадача 2.3.1 осталась ровно тем, чем была:
всё, что добавляется под пункт 2, обязано проходить мимо неё.
"""
from __future__ import annotations

import pytest

import config as config_module
import tasks as tasks_module
from tasks import TaskProfile, UnknownTask


def test_по_умолчанию_подзадача_сдаваемая_сейчас():
    """Умолчание конфига обязано совпадать с попыткой, которую сдаём следующей.

    С 01.08.2026 это 2.3.2: пункт 1 сдан, и забытый ключ --task теперь стоит дорого
    несимметрично — профиль 2.3.1 не создаёт ни детектора, ни реестра, так что дрон
    пролетел бы змейку без единой детекции и привёз ноль из 8 баллов при внешне
    нормальном полёте.
    """
    cfg = config_module.load()
    профиль = tasks_module.from_config(cfg)
    assert профиль.name == "2.3.2"
    assert профиль.detect and профиль.sends_stations and профиль.sends_recon_done


def test_231_ничего_не_ищет_и_не_шлёт():
    профиль = TaskProfile("2.3.1")
    assert not профиль.detect
    assert not профиль.sends_stations
    assert not профиль.sends_recon_done
    assert not профиль.sends_status_update
    assert not профиль.does_dust


def test_232_ищет_станции_но_не_шлёт_статусы():
    """Подзадача 2.3.2 оценивается по числу станций и координатам; статусы —
    это уже 2.3.3, и лишний пакет там ничего не даёт."""
    профиль = TaskProfile("2.3.2")
    assert профиль.detect
    assert профиль.sends_stations
    assert профиль.sends_recon_done
    assert not профиль.sends_status_update


def test_233_добавляет_только_статусы():
    """2.3.3 — чистая надстройка над 2.3.2, всё остальное совпадает."""
    сверх = TaskProfile("2.3.3")
    база = TaskProfile("2.3.2")
    assert сверх.sends_status_update and not база.sends_status_update
    for возможность in ("detect", "station_new", "recon_done", "dust"):
        assert сверх.can(возможность) == база.can(возможность)


def test_234_не_строит_карту_роя():
    """Удаление пыли — самая независимая задача: ни визуализатора, ни протокола
    разведки. Детекция нужна только для наведения."""
    профиль = TaskProfile("2.3.4")
    assert профиль.detect and профиль.does_dust
    assert not профиль.sends_stations
    assert not профиль.sends_recon_done


@pytest.mark.parametrize("ввод,ожидание", [
    ("1", "2.3.1"), ("2", "2.3.2"), ("3", "2.3.3"), ("4", "2.3.4"),
    (" 2.3.2 ", "2.3.2"), ("2.3.1", "2.3.1"),
])
def test_короткие_имена(ввод, ожидание):
    """На площадке набирают «2», а не «2.3.2»; ошибка в номере стоит попытки."""
    assert tasks_module.normalize(ввод) == ожидание


def test_неизвестная_подзадача_падает_громко():
    """Опечатка в --task не должна молча превращаться в «облёт без детекции»:
    попытка была бы потрачена впустую."""
    with pytest.raises(UnknownTask):
        TaskProfile("2.3.9")
    with pytest.raises(UnknownTask):
        TaskProfile("")


def test_ключ_перекрывает_конфиг():
    cfg = config_module.load()
    assert tasks_module.from_config(cfg, "2.3.2").name == "2.3.2"


def test_опечатка_в_возможности_падает():
    """can('станции') вместо can('station_new') иначе читалось бы как «выключено»
    и молча отменило бы отправку пакетов."""
    with pytest.raises(KeyError):
        TaskProfile("2.3.2").can("станции")


def test_все_профили_описывают_все_возможности():
    """Забытый ключ в новом профиле — это KeyError в полёте, а не при импорте."""
    for имя, флаги in tasks_module.TASKS.items():
        отсутствуют = set(tasks_module.CAPABILITIES) - set(флаги)
        assert not отсутствуют, "в профиле %s нет ключей: %s" % (имя, отсутствуют)
        assert флаги["label"], "у профиля %s нет подписи" % имя
