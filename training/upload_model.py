#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Заливка .rknn на борт и регистрация в реестре моделей (арх. custom).

Седьмой шаг конвейера (docs/YOLO_TRAINING.md §7). То же самое можно сделать руками
в веб-интерфейсе борта (http://172.17.49.2:7777/), и там это ровно три поля — но
поле «Архитектура» там по умолчанию предлагает yolov11, а нам нужен `custom`, иначе
ModelContainer постобработает выход сам, ждущим плоскую голову кодом, и наш декодер
сырых веток не увидит ничего. Скрипт закрывает ровно эту ошибку.

    python training/upload_model.py training/rknn/stations_rk3576_int8raw.rknn
    python training/upload_model.py ... --name stations --version 2
    python training/upload_model.py --list                  # что уже в реестре
    python training/upload_model.py --delete stations

Регистрация делается НА БОРТУ: `pioneer_rknn` стоит там, а не у нас, и точный
HTTP-контракт реестра нигде не описан — угадывать его смысла нет, когда есть
готовый ModelRegistry.upload_model(name, version, filepath, arch).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

БОРТ = os.environ.get("DRONE", "172.17.49.2")
ПОЛЬЗОВАТЕЛЬ = os.environ.get("DRONE_USER", "pioneermini")
РАБДИР = os.environ.get("DRONE_DIR", "/home/pioneermini/workspace")

ОПЦИИ = ["-o", "ConnectTimeout=8",
         "-o", "StrictHostKeyChecking=accept-new"]


def адрес(хост, пользователь):
    return "%s@%s" % (пользователь, хост)


def на_борту(код, хост, пользователь):
    """Выполнить питоновский код на борту. python3 -c с одинарными кавычками внутри
    не используем: код содержит пути, и экранирование ломается на первом же пробеле."""
    argv = ["ssh"] + ОПЦИИ + [адрес(хост, пользователь), "python3 -"]
    процесс = subprocess.Popen(argv, stdin=subprocess.PIPE)
    процесс.communicate(код.encode("utf-8"))
    return процесс.returncode


КОД_СПИСКА = """
from pioneer_rknn import ModelRegistry
реестр = ModelRegistry()
print(реестр.list_model())
"""

КОД_ЗАГРУЗКИ = """
from pioneer_rknn import ModelRegistry
реестр = ModelRegistry()
реестр.upload_model(name=%(имя)r, version=%(версия)r, filepath=%(файл)r, arch='custom')
print('в реестре теперь:', реестр.list_model())
"""

КОД_УДАЛЕНИЯ = """
from pioneer_rknn import ModelRegistry
реестр = ModelRegistry()
print(реестр.delete_model(%(имя)r))
print('в реестре теперь:', реестр.list_model())
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("файл", nargs="?", help="путь к .rknn")
    parser.add_argument("--name", default="stations",
                        help="имя модели в реестре; так же зовётся detector.model "
                             "в квадрокоптер/config.yaml")
    parser.add_argument("--version", default="1")
    parser.add_argument("--host", default=БОРТ)
    parser.add_argument("--user", default=ПОЛЬЗОВАТЕЛЬ)
    parser.add_argument("--dir", default=РАБДИР)
    parser.add_argument("--list", action="store_true", help="показать реестр борта")
    parser.add_argument("--delete", default=None, help="удалить модель по имени")
    аргументы = parser.parse_args()

    if аргументы.list:
        return на_борту(КОД_СПИСКА, аргументы.host, аргументы.user)
    if аргументы.delete:
        return на_борту(КОД_УДАЛЕНИЯ % {"имя": аргументы.delete},
                        аргументы.host, аргументы.user)

    if not аргументы.файл:
        parser.error("укажите .rknn (или --list / --delete)")
    if not os.path.exists(аргументы.файл):
        print("нет файла %s" % аргументы.файл)
        return 1
    if not аргументы.файл.endswith(".rknn"):
        print("⚠ это не .rknn — борт принимает только их")

    цель = "%s/models/%s" % (аргументы.dir, os.path.basename(аргументы.файл))
    print("== копирую на борт: %s ==" % цель)
    if subprocess.call(["ssh"] + ОПЦИИ + [адрес(аргументы.host, аргументы.user),
                                          "mkdir -p %s/models" % аргументы.dir]):
        print("борт не отвечает: проверьте Wi-Fi точку PMINI2-* и адрес %s"
              % аргументы.host)
        return 1
    if subprocess.call(["scp"] + ОПЦИИ + [аргументы.файл,
                                          "%s:%s" % (адрес(аргументы.host, аргументы.user),
                                                     цель)]):
        print("scp не отработал")
        return 1

    print("== регистрирую как %r версии %s, архитектура custom =="
          % (аргументы.name, аргументы.version))
    код = на_борту(КОД_ЗАГРУЗКИ % {"имя": аргументы.name, "версия": str(аргументы.version),
                                   "файл": цель},
                   аргументы.host, аргументы.user)
    if код != 0:
        print("регистрация не прошла. Запасной путь — веб-интерфейс борта:")
        print("  http://%s:7777/  (архитектуру указать custom, не yolov11)" % аргументы.host)
        return код

    print("")
    print("Дальше НА БОРТУ, до полёта:")
    print("  python3 tools/diag_model.py --model %s     # девять сырых веток?" % аргументы.name)
    print("  python3 tools/handheld.py                   # рамки над станциями, без взлёта")
    print("И только потом в квадрокоптер/config.yaml: detector.backend: yolo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
