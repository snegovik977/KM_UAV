#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Пульт: одна команда вместо трёх окон, ssh-ключа руками и списка scp.

Штатный порядок работы (RUN.md §9) — это шесть шагов в трёх терминалах: поднять хаб,
залить код, зайти по ssh, запустить main.py, послать takeoff, забрать записи. На полигоне
время идёт по регламенту 1.3 даже когда мы просто набираем команды, а половина ошибок
из таблицы RUN.md §10 — это забытый шаг, а не сломанный код (не снесён __pycache__,
висит второй main.py и держит камеру, хаб поднят не там).

Этот файл делает то же самое сам:

    python launch.py                 ключ -> заливка (с вопросом) -> хаб -> дрон -> команды
    python launch.py --mock          то же самое целиком на своей машине, борт не нужен
    python launch.py --push          только залить и выйти
    python launch.py --setup         развернуть выданную машину: пакеты + тесты
    python launch.py --hub local     хаб на компьютере оператора (по умолчанию на борту)

После старта лаунчер показывает вывод всех процессов с префиксами и ждёт команды:

    t / takeoff   послать пакет «взлёт» (штатный старт подзадачи 2.3.1, не --manual)
    s / state     сводка из хаба: телеметрия и найденные станции
    q / Ctrl+C    посадить дрона, погасить всё, предложить забрать записи

Написан на стандартной библиотеке и без bash: на выданной машине (регламент 1.4 —
свои ноутбуки нельзя) может оказаться Windows, где push_to_drone.sh и setup.sh
не запустятся, а python есть всегда.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))

# Адрес нашего борта, проверен на площадке 31.07.2026. В документации Geoscan
# и у другой команды фигурирует 10.42.0.1 — у нашего экземпляра это не так.
БОРТ = os.environ.get("DRONE", "172.17.49.2")
ПОЛЬЗОВАТЕЛЬ = os.environ.get("DRONE_USER", "pioneermini")
РАБДИР = os.environ.get("DRONE_DIR", "/home/pioneermini/workspace")
ПОРТ_ХАБА = 5001

КЛЮЧ = os.path.join(os.path.expanduser("~"), ".ssh", "id_ed25519")

PY = sys.executable or "python3"


# --------------------------------------------------------------------------- вывод

_печать = threading.Lock()


def скажи(текст):
    with _печать:
        sys.stdout.write(текст + "\n")
        sys.stdout.flush()


def спроси(вопрос, по_умолчанию=True, авто=None):
    """Вопрос да/нет. При --yes/--no и в неинтерактивном запуске не спрашивает."""
    if авто is not None:
        скажи("%s %s" % (вопрос, "да" if авто else "нет"))
        return авто
    if not sys.stdin or not sys.stdin.isatty():
        return по_умолчанию
    подсказка = "[Y/n]" if по_умолчанию else "[y/N]"
    try:
        ответ = input("%s %s " % (вопрос, подсказка)).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if not ответ:
        return по_умолчанию
    return ответ[0] in ("y", "д", "1")


# ----------------------------------------------------------------------------- ssh

def ssh_опции(таймаут=8):
    """Общие ключи ssh/scp. BatchMode не ставим здесь: он нужен только в проверке."""
    return ["-o", "ConnectTimeout=%d" % таймаут,
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=3"]


def адрес_борта(хост=None, пользователь=None):
    return "%s@%s" % (пользователь or ПОЛЬЗОВАТЕЛЬ, хост or БОРТ)


def ssh(команда, хост=None, пользователь=None, tty=False, таймаут=8, batch=False):
    """Собрать argv для ssh. Отдельной функцией — её проверяют тесты."""
    argv = ["ssh"] + ssh_опции(таймаут)
    if batch:
        argv += ["-o", "BatchMode=yes"]
    if tty:
        # -tt форсирует псевдотерминал. Это не косметика: без tty удалённый python
        # переживает обрыв ssh и остаётся висеть, держа камеру (RUN.md §8.3),
        # а с tty в него можно послать Ctrl+C и дрон сядет штатно.
        argv += ["-tt"]
    argv += [адрес_борта(хост, пользователь), команда]
    return argv


def борт_отвечает(хост=None, пользователь=None):
    """Пускает ли борт по ключу прямо сейчас (пароль не спрашивая)."""
    try:
        код = subprocess.call(ssh("true", хост, пользователь, таймаут=5, batch=True),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return False
    return код == 0


def настроить_ключ(хост=None, пользователь=None, авто=None):
    """Сгенерировать ключ, если его нет, и положить публичную часть на борт.

    Пароль борта спрашивается ровно один раз — дальше и лаунчер, и scp, и ssh
    ходят молча. Без этого каждый запуск требует ввода пароля по четыре раза.
    """
    пуб = КЛЮЧ + ".pub"
    if not os.path.exists(пуб):
        if not спроси("SSH-ключа нет. Создать %s?" % КЛЮЧ, True, авто):
            return False
        os.makedirs(os.path.dirname(КЛЮЧ), exist_ok=True)
        код = subprocess.call(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "cv_bpla",
                               "-f", КЛЮЧ])
        if код != 0 or not os.path.exists(пуб):
            скажи("[пульт] ssh-keygen не отработал — дальше придётся вводить пароль руками")
            return False

    with open(пуб, "r", encoding="utf-8") as f:
        ключ = f.read().strip()
    if "'" in ключ:                     # в ed25519-ключе кавычек не бывает, но проверим
        скажи("[пульт] странный ключ, не рискую подставлять его в shell")
        return False

    скажи("[пульт] кладу ключ на борт — пароль борта спросят один раз (geoscan123 / свой)")
    команда = ("mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
               "touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && "
               "grep -qxF '{k}' ~/.ssh/authorized_keys || echo '{k}' >> ~/.ssh/authorized_keys"
               ).format(k=ключ)
    subprocess.call(ssh(команда, хост, пользователь, таймаут=15))
    return борт_отвечает(хост, пользователь)


# -------------------------------------------------------------------------- заливка

def план_заливки(корень=КОРЕНЬ, конфиг=True):
    """Что и куда кладём на борт: список (файлы, поддиректория назначения).

    На борту всё лежит плоско — main.py рядом с protocol/, без директорий с русскими
    именами; main.py умеет находить соседей в обеих раскладках. Хаб и передатчик
    тоже едут на борт: если на ноутбуке нет прав администратора, брандмауэр Windows
    не пропустит входящее на 5001 и команда на взлёт не дойдёт (RUN.md §8.4).
    """
    группы = [
        (sorted(glob.glob(os.path.join(корень, "квадрокоптер", "*.py"))), ""),
        (sorted(glob.glob(os.path.join(корень, "квадрокоптер", "perception", "*.py"))),
         "perception"),
        ([os.path.join(корень, "распределительный хаб", "hub.py"),
          os.path.join(корень, "распределительный хаб", "transmitter.py")], ""),
        (sorted(glob.glob(os.path.join(корень, "распределительный хаб", "protocol", "*.py"))),
         "protocol"),
        ([os.path.join(корень, "tools", имя)
          for имя in ("record.py", "check_sdk.py", "check_axes.py",
                      "check_landing.py")], "tools"),
    ]
    if конфиг:
        группы.insert(1, ([os.path.join(корень, "квадрокоптер", "config.yaml")], ""))
    # Отсутствующих файлов быть не должно, но лучше не валить заливку целиком.
    return [(([f for f in файлы if os.path.exists(f)]), куда) for файлы, куда in группы]


def залить(хост=None, пользователь=None, рабдир=РАБДИР, конфиг=True):
    цель = адрес_борта(хост, пользователь)
    скажи("== Заливка на %s:%s ==" % (цель, рабдир))
    if subprocess.call(ssh("mkdir -p %s/protocol %s/tools %s/perception"
                           % (рабдир, рабдир, рабдир), хост, пользователь)) != 0:
        скажи("[пульт] не смог создать директории на борту")
        return False

    for файлы, куда in план_заливки(конфиг=конфиг):
        if not файлы:
            continue
        назначение = "%s:%s/%s" % (цель, рабдир, куда)
        argv = ["scp"] + ssh_опции() + файлы + [назначение]
        if subprocess.call(argv) != 0:
            скажи("[пульт] scp не отработал: %s" % ", ".join(os.path.basename(f) for f in файлы))
            return False

    # Сносить __pycache__ обязательно каждый раз: иначе Python возьмёт старый .pyc,
    # и час уйдёт на поиски, почему исправленный код ведёт себя по-старому.
    subprocess.call(ssh("cd %s && rm -rf __pycache__ protocol/__pycache__ "
                        "tools/__pycache__ perception/__pycache__"
                        % рабдир, хост, пользователь))
    скажи("[пульт] залито%s" % ("" if конфиг else " (config.yaml на борту не тронут)"))
    return True


def погасить_старое(хост=None, пользователь=None):
    """Убить забытые main.py/hub.py на борту.

    Камера отдаётся через shared memory: второй желающий получает TimeoutError,
    и на площадке это выглядит как «дрон сломался», хотя просто висит вчерашний
    процесс в другом ssh-окне.
    """
    subprocess.call(ssh("pkill -f 'python3 -u main.py' ; pkill -f 'python3 -u hub.py' ; "
                        "pkill -f 'python3 main.py' ; pkill -f 'python3 hub.py' ; true",
                        хост, пользователь),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ------------------------------------------------------------------------ процессы

class Процесс:
    """Запущенная программа с именованным выводом. Логи всех трёх — в одном окне."""

    def __init__(self, имя, argv, env=None, cwd=None, stdin_pipe=False):
        self.имя = имя
        self.argv = argv
        окружение = dict(os.environ)
        окружение["PYTHONUNBUFFERED"] = "1"
        if env:
            окружение.update(env)
        self.proc = subprocess.Popen(
            argv, cwd=cwd or КОРЕНЬ, env=окружение,
            stdin=subprocess.PIPE if stdin_pipe else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True, encoding="utf-8", errors="replace")
        self.поток = threading.Thread(target=self._читать, daemon=True)
        self.поток.start()

    def _читать(self):
        try:
            for строка in self.proc.stdout:
                скажи("[%s] %s" % (self.имя, строка.rstrip()))
        except (ValueError, OSError):
            pass
        скажи("[%s] процесс завершился (код %s)" % (self.имя, self.proc.poll()))

    def жив(self):
        return self.proc.poll() is None

    def ctrl_c(self):
        """Мягкая остановка. Для ssh -tt это единственный способ сказать борту «садись»."""
        if self.proc.stdin and self.жив():
            try:
                self.proc.stdin.write("\x03")
                self.proc.stdin.flush()
                return True
            except (OSError, ValueError):
                pass
        return False

    def стоп(self, ждать=20):
        if not self.жив():
            return
        if not self.ctrl_c():
            self.proc.terminate()
        срок = time.time() + ждать
        while self.жив() and time.time() < срок:
            time.sleep(0.2)
        if self.жив():
            скажи("[пульт] %s не отвечает, добиваю" % self.имя)
            self.proc.kill()


# ----------------------------------------------------------------------------- хаб

def здоровье(url, таймаут=2):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=таймаут) as ответ:
            return json.loads(ответ.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"ok": False, "error": str(e)}


def ждать_хаб(url, секунд=20):
    срок = time.time() + секунд
    while time.time() < срок:
        if здоровье(url).get("ok"):
            return True
        time.sleep(0.5)
    return False


def мой_адрес(хост=None):
    """IP этой машины в сети дрона — его дрон и должен звать хабом.

    Сокет никуда не соединяется по-настоящему (UDP), но ядро выбирает интерфейс,
    через который пошёл бы пакет: ровно тот адрес, который виден борту.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((хост or БОРТ, ПОРТ_ХАБА))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# ------------------------------------------------------------------------ сценарии

def показать_состояние(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/state", timeout=3) as ответ:
            данные = json.loads(ответ.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        скажи("[пульт] хаб не отвечает: %s" % e)
        return
    телеметрия = данные.get("telemetry") or {}
    станции = данные.get("stations") or []
    скажи("[пульт] телеметрия: %s" % (json.dumps(телеметрия, ensure_ascii=False)[:300] or "пусто"))
    скажи("[пульт] станций: %d" % len(станции))
    for с in станции[:10]:
        скажи("        %s" % json.dumps(с, ensure_ascii=False))


def пакет_взлёта(origin_x=0.0, origin_y=0.0, origin_yaw=0.0):
    """Тот же пакет, что шлёт передатчик, — собранный тем же protocol/.

    Пульт не запускает transmitter.py отдельным процессом специально: его телеметрия
    печатается через \\r и перемешивалась бы с выводом дрона в одном окне. А главное —
    формат пакета обязан остаться единым источником правды: в финале вместо передатчика
    встаёт автомобиль, и лаунчер не имеет права знать «свою» версию команды на взлёт.
    """
    sys.path.insert(0, os.path.join(КОРЕНЬ, "распределительный хаб"))
    from protocol import MessageFactory
    return MessageFactory("transmitter").takeoff(
        origin_x=origin_x, origin_y=origin_y, origin_yaw=origin_yaw)


def послать_takeoff(url):
    скажи("[пульт] команда на взлёт -> %s" % url)
    тело = json.dumps(пакет_взлёта()).encode("utf-8")
    запрос = urllib.request.Request(url.rstrip("/") + "/msg", data=тело,
                                    headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(запрос, timeout=5) as ответ:
            ответ.read()
        скажи("[пульт] команда принята хабом")
    except (urllib.error.URLError, OSError) as e:
        скажи("[пульт] хаб не принял команду: %s" % e)


def забрать_записи(хост=None, пользователь=None, рабдир=РАБДИР):
    """Записи с борта — обязательная часть попытки: время полигона тратится и тогда,
    когда мы просто стоим рядом (регламент 1.3), вечером работать не с чем."""
    цель = "%s:%s/flight_*" % (адрес_борта(хост, пользователь), рабдир)
    скажи("[пульт] забираю записи полёта")
    subprocess.call(["scp"] + ssh_опции() + [цель, КОРЕНЬ])


def поднять_локальный_хаб(url, порт, процессы):
    """Поднять хаб на этой машине, если его ещё нет.

    Проверка «а не жив ли уже» — не вежливость: убитый лаунчер (закрытое окно,
    kill -9) оставляет хаб сиротой, второй просто упадёт на занятом порту,
    и это выглядит как «хаб не поднимается».
    """
    if здоровье(url).get("ok"):
        скажи("[пульт] хаб на %s уже поднят — беру его, свой не запускаю" % url)
        return True
    процессы.append(Процесс("хаб", [PY, "-u",
                                    os.path.join(КОРЕНЬ, "распределительный хаб", "hub.py"),
                                    "--port", str(порт)]))
    return ждать_хаб(url, 20)


def поднять_визуализатор(url, процессы, headless=False):
    """Карта станций на этой машине. Без неё 8 баллов подзадачи 2.3.2 не даются:
    критерий прямо требует, чтобы визуализатор отображал станции в реальном времени."""
    argv = [PY, "-u", os.path.join(КОРЕНЬ, "распределительный хаб", "visualizer.py"),
            "--hub", url]
    if headless:
        argv.append("--headless")
    процессы.append(Процесс("карта", argv))


# Что нужно на машине разработки и зачем. Бортовому ПО из этого не нужно ничего:
# транспорт написан на стандартной библиотеке специально (requirements.txt).
ПАКЕТЫ = (
    ("numpy",  "numpy",         "нужен перцепции и заглушке камеры",          True),
    ("cv2",    "opencv-python", "нужен записи попыток и перцепции",           True),
    ("pytest", "pytest",        "нужен только для прогона тестов",            True),
    ("flask",  "flask",         "необязателен: хаб умеет работать на stdlib", False),
    ("yaml",   "PyYAML",        "необязателен: config.py читает YAML сам",    False),
)


def сценарий_настройки(аргументы):
    """Развернуть репозиторий на выданной машине.

    По регламенту 1.4 личные компьютеры на площадке запрещены, значит всё поднимается
    с нуля на чужом железе, и обычно под Windows — поэтому проверка живёт здесь,
    а не в bash-скрипте.
    """
    скажи("== Окружение ==")
    скажи("Python: %s" % sys.version.split()[0])
    скажи("Репозиторий: %s" % КОРЕНЬ)

    скажи("")
    скажи("== Библиотеки ==")
    ставить = []
    for модуль, пакет, зачем, обязателен in ПАКЕТЫ:
        есть = subprocess.call([PY, "-c", "import %s" % модуль],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
        скажи("  %s  %-7s (%s)" % ("ЕСТЬ" if есть else "НЕТ ", модуль, зачем))
        if not есть and обязателен:
            ставить.append(пакет)

    if ставить:
        if спроси("Доставить: %s ?" % ", ".join(ставить), True, аргументы.авто):
            # Без venv: на борту в виртуальном окружении pioneer_sdk2 не работает,
            # и одна и та же команда должна годиться и там, и на выданной машине.
            subprocess.call([PY, "-m", "pip", "install", "--user"] + ставить)
        else:
            скажи("[пульт] без %s часть проверок не пройдёт" % ", ".join(ставить))

    скажи("")
    скажи("== Тесты ==")
    if subprocess.call([PY, "-c", "import pytest"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        скажи("  pytest не установлен, пропускаю")
        return 1
    код = subprocess.call([PY, "-m", "pytest", "tests", "-q"], cwd=КОРЕНЬ)
    if код != 0:
        скажи("")
        скажи("ТЕСТЫ УПАЛИ — на полигон не идём, сначала чиним")
        return 1

    скажи("")
    скажи("Готово. Дальше:")
    скажи("  python launch.py --mock      прогон без дрона")
    скажи("  python launch.py             борт: заливка и запуск (RUN.md §0)")
    return 0


def сценарий_заглушки(аргументы):
    """Всё на своей машине: хаб + дрон в mock. Заменяет три окна из RUN.md §4."""
    url = "http://127.0.0.1:%d" % аргументы.port
    процессы = []
    if not поднять_локальный_хаб(url, аргументы.port, процессы):
        скажи("[пульт] хаб не поднялся, дальше идти незачем")
        for п in процессы:
            п.стоп()
        return 1
    скажи("[пульт] хаб жив: %s" % url)
    if аргументы.viz:
        поднять_визуализатор(url, процессы, аргументы.viz_headless)

    argv = [PY, "-u", os.path.join(КОРЕНЬ, "квадрокоптер", "main.py"), "--hub", url]
    if аргументы.task:
        argv += ["--task", аргументы.task]
    if аргументы.no_camera:
        argv.append("--no-camera")
    if аргументы.no_record:
        argv.append("--no-record")
    процессы.append(Процесс("дрон", argv, env={"PIONEER_MOCK": "1"}))
    return пульт(процессы, url, аргументы, борт=False)


def сценарий_борта(аргументы):
    хост, пользователь = аргументы.drone, аргументы.user
    # --------------------------------------------------------------- связь и ключ
    if not борт_отвечает(хост, пользователь):
        скажи("[пульт] борт %s по ключу не пускает" % адрес_борта(хост, пользователь))
        if not настроить_ключ(хост, пользователь, аргументы.авто):
            скажи("[пульт] без ключа дальше будет пароль на каждый шаг. "
                  "Проверьте: подключены ли к сети PMINI2-*, тот ли адрес борта "
                  "(наш — 172.17.49.2, а не 10.42.0.1 из документации)")
            if not спроси("Продолжить всё равно?", False, аргументы.авто):
                return 1
    else:
        скажи("[пульт] борт %s на связи" % адрес_борта(хост, пользователь))

    # ------------------------------------------------------------------- заливка
    if спроси("Перезалить код на борт?", True, аргументы.авто):
        if not залить(хост, пользователь, аргументы.dir, конфиг=not аргументы.keep_config):
            return 1
    if аргументы.push:
        return 0

    погасить_старое(хост, пользователь)

    процессы = []
    # ----------------------------------------------------------------------- хаб
    if аргументы.hub == "drone":
        url = "http://%s:%d" % (хост, аргументы.port)
        url_для_дрона = "http://127.0.0.1:%d" % аргументы.port
        процессы.append(Процесс(
            "хаб", ssh("cd %s && exec python3 -u hub.py --port %d" % (аргументы.dir, аргументы.port),
                       хост, пользователь, tty=True), stdin_pipe=True))
    elif аргументы.hub == "local":
        url = url_для_дрона = "http://%s:%d" % (мой_адрес(хост), аргументы.port)
        поднять_локальный_хаб(url, аргументы.port, процессы)
        скажи("[пульт] хаб на этой машине, дрон будет звать его по %s" % url)
        скажи("[пульт] если дрон получит timed out — это брандмауэр Windows, "
              "перезапуститесь с --hub drone (RUN.md §8.4)")
    else:                                  # --hub none: хаб уже поднят кем-то другим
        url = url_для_дрона = аргументы.hub_url or ("http://%s:%d" % (хост, аргументы.port))
        скажи("[пульт] свой хаб не поднимаю, работаю с %s" % url)

    if not ждать_хаб(url, 25):
        скажи("[пульт] хаб %s не отвечает" % url)
        for п in процессы:
            п.стоп()
        return 1
    скажи("[пульт] хаб жив: %s   (карта: %s/state)" % (url, url))
    if аргументы.viz:
        поднять_визуализатор(url, процессы, аргументы.viz_headless)

    # ---------------------------------------------------------------------- дрон
    ключи = ["--hub", url_для_дрона]
    if аргументы.task:
        ключи += ["--task", аргументы.task]
    if аргументы.manual:
        # На зачётной попытке нельзя: 3 балла из 5 дают именно за старт по пакету.
        скажи("[пульт] ВНИМАНИЕ: --manual — только отладка, на зачётной попытке "
              "дрон обязан стартовать по команде от передатчика (RUN.md §9.1)")
        ключи.append("--manual")
    if аргументы.no_camera:
        ключи.append("--no-camera")
    if аргументы.no_record:
        ключи.append("--no-record")
    команда = "cd %s && exec python3 -u main.py %s" % (аргументы.dir, " ".join(ключи))
    процессы.append(Процесс("дрон", ssh(команда, хост, пользователь, tty=True), stdin_pipe=True))

    return пульт(процессы, url, аргументы, борт=True)


def читалка_stdin():
    """Ввод пользователя отдельным потоком.

    input() в главном цикле блокировал бы до нажатия Enter, и завершение миссии
    пульт замечал бы только после того, как оператор что-нибудь наберёт.
    """
    очередь = queue.Queue()

    def читать():
        try:
            for строка in sys.stdin:
                очередь.put(строка)
        except (ValueError, OSError):
            pass
        очередь.put(None)

    threading.Thread(target=читать, daemon=True).start()
    return очередь


def пульт(процессы, url, аргументы, борт):
    """Показывать логи и принимать команды, пока не скажут стоп.

    Главный процесс — дрон, он всегда последний в списке: когда миссия закончилась,
    держать окно открытым незачем, хаб гасится следом.
    """
    дрон = процессы[-1]
    скажи("")
    скажи("== Пульт ==  t — взлёт   s — состояние   q — посадить и выйти  (Ctrl+C тоже) ==")
    скажи("")

    if аргументы.takeoff:
        time.sleep(2.0)
        послать_takeoff(url)

    интерактив = bool(sys.stdin) and sys.stdin.isatty()
    ввод = читалка_stdin() if интерактив else None
    try:
        while True:
            if not дрон.жив():
                скажи("[пульт] дрон отработал, гашу остальное")
                break
            if ввод is None:
                time.sleep(0.5)
                continue
            try:
                строка = ввод.get(timeout=0.5)
            except queue.Empty:
                continue
            if строка is None:
                интерактив = False
                ввод = None
                continue
            команда = строка.strip().lower()
            if команда in ("t", "takeoff", "взлёт", "взлет"):
                послать_takeoff(url)
            elif команда in ("s", "state", "состояние"):
                показать_состояние(url)
            elif команда in ("q", "quit", "exit", "стоп"):
                break
            elif команда:
                скажи("[пульт] не понял: t — взлёт, s — состояние, q — выход")
    except KeyboardInterrupt:
        скажи("")
        скажи("[пульт] Ctrl+C — сажаю дрона")

    # Гасим в обратном порядке: сначала дрон (ему нужно сесть), потом хаб.
    for п in reversed(процессы):
        п.стоп()

    if борт and спроси("Забрать записи полёта с борта?", True, аргументы.авто):
        забрать_записи(аргументы.drone, аргументы.user, аргументы.dir)
        скажи("[пульт] разбор: %s tools/record.py flight_*.jsonl" % os.path.basename(PY))
    return 0


# ---------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description="Пульт: заливка на борт и запуск всей системы одной командой")
    parser.add_argument("--mock", action="store_true",
                        help="всё на своей машине, борт не нужен (заглушка)")
    parser.add_argument("--task", default=None,
                        help="подзадача регламента: 2.3.1 | 2.3.2 | 2.3.3 | 2.3.4 "
                             "(или 1..4). По умолчанию — mission.task из config.yaml")
    parser.add_argument("--push", action="store_true", help="только залить код и выйти")
    parser.add_argument("--setup", action="store_true",
                        help="развернуть машину: проверить пакеты, доставить, прогнать тесты")
    parser.add_argument("--hub", choices=("drone", "local", "none"), default="drone",
                        help="где поднимать хаб (по умолчанию на борту — не мешает брандмауэр)")
    parser.add_argument("--hub-url", default=None, dest="hub_url",
                        help="адрес уже поднятого хаба, вместе с --hub none")
    parser.add_argument("--takeoff", action="store_true",
                        help="послать команду на взлёт сразу после старта")
    parser.add_argument("--manual", action="store_true",
                        help="дрон взлетает сам, не дожидаясь пакета (только отладка)")
    parser.add_argument("--viz", action="store_true",
                        help="поднять визуализатор (карту станций). Для подзадач 2.3.2 "
                             "и 2.3.3 он обязателен: баллы дают за отображение "
                             "станций в реальном времени")
    parser.add_argument("--viz-headless", action="store_true", dest="viz_headless",
                        help="визуализатор без окна, только лог принятых станций")
    parser.add_argument("--no-camera", action="store_true", dest="no_camera")
    parser.add_argument("--no-record", action="store_true", dest="no_record")
    parser.add_argument("--keep-config", action="store_true", dest="keep_config",
                        help="не перезаписывать config.yaml на борту (правили на месте)")
    parser.add_argument("--drone", default=БОРТ, help="адрес борта")
    parser.add_argument("--user", default=ПОЛЬЗОВАТЕЛЬ, help="ssh-логин борта")
    parser.add_argument("--dir", default=РАБДИР, help="рабочая директория на борту")
    parser.add_argument("--port", type=int, default=ПОРТ_ХАБА)
    parser.add_argument("-y", "--yes", action="store_true", help="на все вопросы «да»")
    parser.add_argument("-n", "--no", action="store_true", help="на все вопросы «нет»")
    аргументы = parser.parse_args()
    аргументы.авто = True if аргументы.yes else (False if аргументы.no else None)

    if аргументы.setup:
        return сценарий_настройки(аргументы)
    if аргументы.mock:
        return сценарий_заглушки(аргументы)
    if not shutil.which("ssh"):
        скажи("[пульт] ssh не найден. Windows: «Дополнительные компоненты» -> "
              "«Клиент OpenSSH». Прогон без борта — python launch.py --mock")
        return 1
    return сценарий_борта(аргументы)


if __name__ == "__main__":
    sys.exit(main())
