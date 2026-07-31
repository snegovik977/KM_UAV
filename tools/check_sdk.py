#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Диагностика борта, день 1. Без полёта, без arm().

Один запуск закрывает список проверок из docs/organizer_handouts.md §4.3 и
docs/lessons_from_archipelago.md §9. Скрипт отвечает на вопросы, по которым три наших
источника (сайт Geoscan, раздатка организаторов, полевой опыт чужой команды) расходятся:

  - как реально называются спорные методы (get_dist_sensor_data / get_ranger_data,
    point_reached, аргументы get_local_position_lps);
  - до какого угла реально доезжает ServoCamera.set_angle (-90 или -80);
  - какое разрешение у get_cv_frame() и в каком формате кадр;
  - какой SoC на борту (rk3566 / rk3576) — от этого зависит target_platform при
    конвертации .rknn;
  - на каком порту живёт ImageViewer (8089 или 8889);
  - есть ли в системном Python requests / flask.

Ответы переносятся руками в квадрокоптер/config.yaml — там они помечены
«# НЕ ПРОВЕРЕНО (check_sdk.py)».

Запуск на борту:
    python3 check_sdk.py                # с подключением к автопилоту (только чтение)
    python3 check_sdk.py --no-connect   # только интроспекция классов
    python3 check_sdk.py --no-servo     # не трогать сервопривод камеры

Отчёт печатается в консоль и пишется в check_sdk_report.txt рядом со скриптом.
Скрипт никогда не падает: каждая проверка изолирована, недоступная — помечается ПРОПУСК.
"""
from __future__ import annotations

import argparse
import inspect
import io
import os
import platform
import sys
import traceback

REPORT_NAME = "check_sdk_report.txt"

# Методы, вокруг которых есть расхождение источников. Проверяем поимённо, а не «на глаз»
# по общему списку: важно не только что метод есть, но и какая у него сигнатура.
DISPUTED = {
    "Pioneer": [
        "arm", "disarm", "takeoff", "land", "rtl", "is_landed",
        "go_to_local_point", "go_to_local_point_body_fixed", "set_yaw",
        "point_reached",                       # есть только в раздатке организаторов
        "get_local_position_lps",              # с аргументом update или без?
        "get_nav_status_lps", "get_orientation", "get_altitude",
        "get_battery_status",
        "get_dist_sensor_data",                # раздатка организаторов
        "get_ranger_data",                     # сайт Geoscan
        "subscribe", "unsubscribe", "close_connection",
    ],
    "Camera": ["get_cv_frame", "stop"],
    "ServoCamera": ["set_angle"],
    "ImageViewer": ["imshow", "close"],
}


class Report:
    """Копит текст отчёта: всё уходит и в консоль, и в файл."""

    def __init__(self):
        self.lines = []

    def line(self, text=""):
        print(text)
        self.lines.append(text)

    def section(self, title):
        self.line()
        self.line("=" * 78)
        self.line(title)
        self.line("=" * 78)

    def fail(self, what, exc):
        self.line("  ПРОПУСК: %s -> %s: %s" % (what, type(exc).__name__, exc))

    def save(self, path):
        try:
            with io.open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.lines) + "\n")
            print("\nОтчёт сохранён: %s" % path)
        except Exception as e:
            print("\nНе удалось сохранить отчёт: %s" % e)


def sig_of(obj, name):
    """Сигнатура метода в виде строки или пометка, что метода нет."""
    fn = getattr(obj, name, None)
    if fn is None:
        return "ОТСУТСТВУЕТ"
    try:
        return name + str(inspect.signature(fn))
    except (TypeError, ValueError):
        return name + "(...)  # сигнатура не читается (C-расширение?)"


def public_methods(cls):
    """Все публичные методы класса с сигнатурами — замена help() в читаемом виде."""
    out = []
    for name in sorted(dir(cls)):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(cls, name)
        except Exception:
            continue
        if callable(attr):
            out.append("    " + sig_of(cls, name))
    return out


# ---------------------------------------------------------------- проверки

def check_environment(rep):
    rep.section("1. Окружение")
    rep.line("  python:   %s" % sys.version.replace("\n", " "))
    rep.line("  platform: %s" % platform.platform())
    rep.line("  machine:  %s" % platform.machine())
    rep.line("  cwd:      %s" % os.getcwd())

    # SoC решает target_platform при конвертации .rknn: rk3566 или rk3576.
    # Полевой опыт чужой команды на этом же дроне — rk3576, документация подразумевала rk3566.
    for path in ("/proc/device-tree/compatible", "/sys/firmware/devicetree/base/compatible"):
        try:
            with io.open(path, "rb") as f:
                raw = f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
            rep.line("  SoC (%s): %s" % (path, raw))
            break
        except Exception as e:
            rep.fail(path, e)


def check_imports(rep):
    rep.section("2. Библиотеки")
    for name in ("pioneer_sdk2", "pioneer_rknn", "cv2", "numpy", "yaml", "requests", "flask"):
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "версия не указана")
            path = getattr(mod, "__file__", "?")
            rep.line("  ЕСТЬ  %-14s %-12s %s" % (name, ver, path))
        except Exception as e:
            rep.line("  НЕТ   %-14s %s: %s" % (name, type(e).__name__, e))
    rep.line()
    rep.line("  Примечание: если нет requests/flask — это ожидаемо и нам не мешает.")
    rep.line("  Транспорт дрона написан на stdlib (urllib), ставить на борт ничего не нужно.")


def check_signatures(rep):
    rep.section("3. Сигнатуры классов SDK")
    try:
        import pioneer_sdk2
    except Exception as e:
        rep.fail("import pioneer_sdk2", e)
        return

    for cls_name, methods in DISPUTED.items():
        cls = getattr(pioneer_sdk2, cls_name, None)
        rep.line()
        rep.line("  --- %s ---" % cls_name)
        if cls is None:
            rep.line("    КЛАСС ОТСУТСТВУЕТ в pioneer_sdk2")
            continue
        rep.line("    конструктор: %s" % sig_of(cls, "__init__"))
        rep.line("    спорные методы:")
        for m in methods:
            rep.line("      %s" % sig_of(cls, m))
        rep.line("    все публичные методы:")
        for line in public_methods(cls):
            rep.line("    " + line)

    # Перечисления, которые используются в коде по именам.
    for enum_name in ("CameraType", "NavSystem", "Event", "ServoPriority"):
        enum = getattr(pioneer_sdk2, enum_name, None)
        if enum is None:
            rep.line("\n  %s: ОТСУТСТВУЕТ" % enum_name)
            continue
        members = [n for n in dir(enum) if not n.startswith("_")]
        rep.line("\n  %s: %s" % (enum_name, ", ".join(sorted(members))))

    rep.line()
    rep.line("  Прочее в модуле pioneer_sdk2:")
    rep.line("    " + ", ".join(sorted(n for n in dir(pioneer_sdk2) if not n.startswith("_"))))


def check_rknn(rep):
    rep.section("4. pioneer_rknn")
    try:
        import pioneer_rknn
    except Exception as e:
        rep.fail("import pioneer_rknn", e)
        return
    rep.line("  Содержимое модуля: %s"
             % ", ".join(sorted(n for n in dir(pioneer_rknn) if not n.startswith("_"))))
    for cls_name in ("Yolo", "ModelContainer", "ModelRegistry"):
        cls = getattr(pioneer_rknn, cls_name, None)
        rep.line()
        rep.line("  --- %s ---" % cls_name)
        if cls is None:
            rep.line("    ОТСУТСТВУЕТ")
            continue
        rep.line("    конструктор: %s" % sig_of(cls, "__init__"))
        # arch говорит, какие архитектуры контейнер постобработает сам. Нам нужен custom:
        # с ним ModelContainer отдаёт сырые выходы, а decode мы делаем на CPU (пункт 2 плана).
        rep.line("    arch: %r" % getattr(cls, "arch", "нет атрибута"))
        for line in public_methods(cls):
            rep.line("    " + line)

    # Что уже зарегистрировано в реестре моделей на :7777.
    reg_cls = getattr(pioneer_rknn, "ModelRegistry", None)
    if reg_cls is not None:
        try:
            reg = reg_cls()
            rep.line()
            rep.line("  Модели в реестре: %r" % (reg.list_model(),))
        except Exception as e:
            rep.fail("ModelRegistry().list_model()", e)


def check_telemetry(rep):
    """Подключение к автопилоту и чтение телеметрии. Ничего не armит и не взлетает."""
    rep.section("5. Телеметрия (только чтение, без arm)")
    try:
        from pioneer_sdk2 import Pioneer
    except Exception as e:
        rep.fail("import Pioneer", e)
        return

    pioneer = None
    try:
        pioneer = Pioneer()
        rep.line("  Подключение установлено")
    except Exception as e:
        rep.fail("Pioneer()", e)
        return

    try:
        # get_local_position_lps: сайт Geoscan даёт метод без аргументов, пример
        # организаторов зовёт get_local_position_lps(True). Пробуем оба варианта.
        for call, label in (
            (lambda: pioneer.get_local_position_lps(), "get_local_position_lps()"),
            (lambda: pioneer.get_local_position_lps(True), "get_local_position_lps(True)"),
        ):
            try:
                rep.line("  %-32s -> %r" % (label, call()))
            except Exception as e:
                rep.line("  %-32s -> %s: %s" % (label, type(e).__name__, e))

        simple = [
            "get_orientation", "get_altitude", "get_battery_status",
            "get_nav_status_lps", "get_nav_system", "get_satellites_count",
            "is_landed", "point_reached",
            # Спор о названии дальномера: раздатка организаторов vs сайт Geoscan.
            # Тот из двух, что вернёт число, и станет источником высоты для h_min (пункт 4).
            "get_dist_sensor_data", "get_ranger_data",
        ]
        for name in simple:
            fn = getattr(pioneer, name, None)
            if fn is None:
                rep.line("  %-32s -> МЕТОДА НЕТ" % (name + "()"))
                continue
            try:
                rep.line("  %-32s -> %r" % (name + "()", fn()))
            except Exception as e:
                rep.line("  %-32s -> %s: %s" % (name + "()", type(e).__name__, e))
    finally:
        try:
            pioneer.close_connection()
            rep.line("  Соединение закрыто")
        except Exception as e:
            rep.fail("close_connection()", e)


def check_servo(rep):
    """Пробы set_angle: где именно начинается ValueError.

    Разница -90 против -80 — это ~10° постоянного наклона камеры, то есть ~35 см
    смещения на высоте 2 м. Больше всего порога «<30 см» из подзадачи 2.3.2, поэтому
    фактический угол обязан лежать в config.yaml рядом с матрицей R_mount, измеренной
    именно при нём (docs/lessons_from_archipelago.md §1.1).
    """
    rep.section("6. ServoCamera.set_angle — реальный диапазон")
    try:
        from pioneer_sdk2 import ServoCamera
    except Exception as e:
        rep.fail("import ServoCamera", e)
        return
    try:
        servo = ServoCamera()
    except Exception as e:
        rep.fail("ServoCamera()", e)
        return

    for angle in (-95, -90, -85, -80, -75, -45, 0, 30, 35):
        try:
            result = servo.set_angle(angle)
            rep.line("  set_angle(%4d) -> ПРИНЯТ, вернул %r" % (angle, result))
        except Exception as e:
            rep.line("  set_angle(%4d) -> %s: %s" % (angle, type(e).__name__, e))

    rep.line()
    rep.line("  ГЛАЗАМИ: минимальный принятый угол — это надир или камера всё ещё наклонена?")
    rep.line("  Записать фактический угол в config.yaml -> camera.servo_angle.")
    try:
        servo.set_angle(0)
        rep.line("  Камера возвращена в 0")
    except Exception as e:
        rep.fail("set_angle(0)", e)


def check_camera(rep):
    """Один кадр с MAIN. OPT не открываем никогда — это камера оптического потока,
    второй потребитель сажает позиционирование (docs/lessons_from_archipelago.md §1.2).
    """
    rep.section("7. Камера MAIN — один кадр")
    try:
        from pioneer_sdk2 import Camera, CameraType
    except Exception as e:
        rep.fail("import Camera", e)
        return

    camera = None
    try:
        camera = Camera(camera_type=CameraType.MAIN)
        frame = camera.get_cv_frame(timeout=5.0)
        if frame is None:
            rep.line("  get_cv_frame() вернул None")
        else:
            rep.line("  shape: %r" % (getattr(frame, "shape", "?"),))
            rep.line("  dtype: %r" % (getattr(frame, "dtype", "?"),))
            rep.line("  Разрешение нужно для пересчёта площади объекта в м² (пункт 2 плана).")
            try:
                import cv2
                out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_sdk_frame.png")
                cv2.imwrite(out, frame)
                rep.line("  Кадр сохранён: %s (посмотреть глазами — что реально видит камера)" % out)
            except Exception as e:
                rep.fail("сохранение кадра", e)
    except Exception as e:
        rep.fail("Camera(MAIN).get_cv_frame()", e)
    finally:
        # Камера отдаётся через shared memory: не остановить — следующий запуск получит
        # TimeoutError («shm_open failed»).
        if camera is not None:
            try:
                camera.stop()
                rep.line("  camera.stop() выполнен")
            except Exception as e:
                rep.fail("camera.stop()", e)


def check_viewer_ports(rep):
    """8089 (сайт Geoscan) или 8889 (раздатка организаторов) — от этого зависит,
    найдём ли мы отладочную трансляцию, единственный способ видеть детекцию в полёте."""
    rep.section("8. ImageViewer — порт трансляции")
    import urllib.request

    for port in (8089, 8889):
        for path in ("/", "/video"):
            url = "http://127.0.0.1:%d%s" % (port, path)
            try:
                resp = urllib.request.urlopen(url, timeout=2.0)
                rep.line("  ОТВЕЧАЕТ %-34s код %s" % (url, resp.getcode()))
                resp.close()
            except Exception as e:
                rep.line("  молчит   %-34s %s: %s" % (url, type(e).__name__, e))

    rep.line()
    rep.line("  Порт мог не ответить просто потому, что ImageViewer сейчас никем не открыт.")
    rep.line("  Достоверная проверка — запустить main.py с камерой и открыть оба порта в браузере.")


def main():
    parser = argparse.ArgumentParser(description="Диагностика борта Pioneer Mini 2 (без полёта)")
    parser.add_argument("--no-connect", action="store_true",
                        help="не подключаться к автопилоту (только интроспекция классов)")
    parser.add_argument("--no-servo", action="store_true",
                        help="не трогать сервопривод камеры")
    parser.add_argument("--no-camera", action="store_true",
                        help="не открывать камеру")
    args = parser.parse_args()

    rep = Report()
    rep.line("Диагностика борта Pioneer Mini 2 — проверки дня 1, без полёта")
    rep.line("Результаты переносятся в квадрокоптер/config.yaml (метки «НЕ ПРОВЕРЕНО»)")

    steps = [check_environment, check_imports, check_signatures, check_rknn]
    if not args.no_connect:
        steps.append(check_telemetry)
    if not args.no_servo:
        steps.append(check_servo)
    if not args.no_camera:
        steps.append(check_camera)
    steps.append(check_viewer_ports)

    for step in steps:
        try:
            step(rep)
        except Exception:
            rep.line("  НЕОЖИДАННАЯ ОШИБКА в %s:" % step.__name__)
            for line in traceback.format_exc().splitlines():
                rep.line("    " + line)

    rep.section("Что делать с этим отчётом")
    rep.line("  1. Заполнить квадрокоптер/config.yaml там, где стоит «НЕ ПРОВЕРЕНО».")
    rep.line("  2. Поправить квадрокоптер/mock_pioneer.py под реальные сигнатуры.")
    rep.line("  3. Оси и единицы yaw этим скриптом не проверить — нужен tools/check_axes.py.")

    rep.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), REPORT_NAME))


if __name__ == "__main__":
    main()
