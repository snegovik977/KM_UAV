# -*- coding: utf-8 -*-
"""Профили подзадач регламента: что бортовое ПО делает, а чего не делает.

Одна и та же программа сдаёт несколько подзадач 2.3.x подряд, и они отличаются не
маршрутом, а тем, какие ветки включены. Профиль — единственное место, где это записано,
чтобы «а в этой попытке мы статусы шлём?» не выяснялось по трём разным флагам в разных
файлах прямо перед вылетом.

    2.3.1  взлёт по команде, облёт, посадка. Перцепция крутится (кадр -> HUD ->
           трансляция -> запись), но детекции нет вообще: ложное срабатывание не может
           испортить попытку, которой оно не нужно.
    2.3.2  то же плюс детекция станций, локализация, station_new и recon_done.
    2.3.3  то же плюс отправка status_update при смене статуса станции.
    2.3.4  удаление пыли: детекция нужна для наведения, но карта роя не строится —
           станции никому не передаются (задел пункта 4 плана).

Почему профили в коде, а не в config.yaml: это структура решения, а не настройка под
расстановку. В конфиге лежит только имя выбранной задачи (mission.task), и требование
регламента 2.5 «никаких зашитых чисел» относится к порогам и координатам, а не к тому,
какая подзадача сдаётся. Плюс самодельный YAML-парсер (config.py) не умеет ключи вида
"2.3.1" в кавычках — заводить их там пришлось бы через костыль.
"""
from __future__ import annotations

# Порядок ключей = порядок сдачи по docs/DRONE_PLAN.md.
TASKS = {
    "2.3.1": {
        "label": "облёт территории по команде",
        "detect": False,
        "station_new": False,
        "recon_done": False,
        "status_update": False,
        "dust": False,
    },
    "2.3.2": {
        "label": "обнаружение электростанций",
        "detect": True,
        "station_new": True,
        "recon_done": True,
        "status_update": False,
        "dust": False,
    },
    "2.3.3": {
        "label": "визуальное определение статуса",
        "detect": True,
        "station_new": True,
        "recon_done": True,
        "status_update": True,
        "dust": False,
    },
    "2.3.4": {
        # Задел пункта 4 плана: манёвр обдува ещё не написан, состояние DUST автомат
        # проходит насквозь. Профиль заведён сразу, чтобы пункт 4 не потребовал править
        # разбор аргументов и конфиг.
        "label": "удаление пыли с солнечной батареи",
        "detect": True,
        "station_new": False,
        "recon_done": False,
        "status_update": False,
        "dust": True,
    },
}

# Синонимы: на площадке проще набрать «1», чем «2.3.1», а ошибиться в номере — дорого.
_ПСЕВДОНИМЫ = {"1": "2.3.1", "2": "2.3.2", "3": "2.3.3", "4": "2.3.4"}

NAMES = tuple(TASKS)
DEFAULT = "2.3.1"

# Возможности, которые есть у каждого профиля. Опечатка в ключе иначе читалась бы
# как «выключено» и молча отменяла бы отправку пакетов.
CAPABILITIES = ("detect", "station_new", "recon_done", "status_update", "dust")


class UnknownTask(ValueError):
    """Запрошена подзадача, которой нет в TASKS."""


def normalize(name):
    """Привести имя задачи к каноническому виду: '2', '2.3.2', ' 2.3.2 ' -> '2.3.2'."""
    text = str(name).strip()
    text = _ПСЕВДОНИМЫ.get(text, text)
    if text not in TASKS:
        raise UnknownTask("неизвестная подзадача %r; известные: %s"
                          % (name, ", ".join(NAMES)))
    return text


class TaskProfile(object):
    """Что включено в текущей подзадаче. Только чтение — профиль не меняется в полёте."""

    def __init__(self, name):
        self.name = normalize(name)
        self._flags = TASKS[self.name]

    @property
    def label(self):
        return self._flags["label"]

    def can(self, capability):
        if capability not in CAPABILITIES:
            raise KeyError("нет такой возможности профиля: %r (есть: %s)"
                           % (capability, ", ".join(CAPABILITIES)))
        return bool(self._flags[capability])

    # Читаемые сокращения для мест, где условие стоит в горячем коде.
    @property
    def detect(self):
        return self.can("detect")

    @property
    def sends_stations(self):
        return self.can("station_new")

    @property
    def sends_recon_done(self):
        return self.can("recon_done")

    @property
    def sends_status_update(self):
        return self.can("status_update")

    @property
    def does_dust(self):
        return self.can("dust")

    def __str__(self):
        включено = [c for c in CAPABILITIES if self._flags[c]]
        return "%s (%s): %s" % (self.name, self.label,
                                ", ".join(включено) if включено else "только полёт")

    def __repr__(self):
        return "TaskProfile(%r)" % (self.name,)


def from_config(cfg, override=None):
    """Профиль по config.yaml -> mission.task; override (ключ --task) важнее конфига."""
    name = override if override else cfg["mission"].get("task", DEFAULT)
    return TaskProfile(name)
