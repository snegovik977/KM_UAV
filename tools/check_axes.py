#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка осей go_to_local_point и единиц yaw. ЭТО ПОЛЁТ — нужен полигон.

Единственное расхождение источников, которое нельзя закрыть интроспекцией:

  - сайт Geoscan подразумевает X вперёд, Y влево;
  - раздатка организаторов прямо пишет «ось y направлена вперёд, ось x — вправо»
    (docs/organizer_handouts.md §3.1);
  - единицы yaw противоречат сами себе внутри одного PDF: в комментарии к примеру
    «в радианах», в справочной карточке метода «в градусах».

Ошибка по осям не падает с исключением — она разворачивает всю змейку на 90°, и это
видно только в воздухе. Ошибка по yaw в радианах против градусов раскрутит дрон
на несколько оборотов.

Полёт короткий: взлёт на 1 м, три смещения по 1 м, разворот, посадка. ~60 секунд.
Результат руками переносится в квадрокоптер/config.yaml -> axes.

Запуск (дрон на площадке, вокруг свободно не меньше 2 м):
    python3 check_axes.py
    python3 check_axes.py --height 1.2 --step 0.8

СМОТРЕТЬ ГЛАЗАМИ и записывать, куда физически уехал дрон. Скрипт печатает подсказку
перед каждым манёвром и показания LPS после него — но истина здесь наблюдение, а не
телеметрия: LPS выдаёт координаты в той же спорной СК.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

REPORT_NAME = "check_axes_report.txt"


class Report:
    def __init__(self):
        self.lines = []

    def line(self, text=""):
        print(text)
        sys.stdout.flush()
        self.lines.append(text)

    def save(self, path):
        try:
            with io.open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.lines) + "\n")
            print("\nОтчёт сохранён: %s" % path)
        except Exception as e:
            print("\nНе удалось сохранить отчёт: %s" % e)


def read_state(pioneer):
    """Позиция и ориентация как их отдаёт SDK, без пересчётов."""
    pos, ori = None, None
    try:
        pos = pioneer.get_local_position_lps()
    except Exception as e:
        pos = "ошибка: %s" % e
    try:
        ori = pioneer.get_orientation()
    except Exception as e:
        ori = "ошибка: %s" % e
    return pos, ori


def wait_point(pioneer, timeout, rep):
    """Ожидание прихода в точку. point_reached() — из примера организаторов;
    если метода нет, ждём по таймауту (для короткого перелёта этого достаточно)."""
    fn = getattr(pioneer, "point_reached", None)
    deadline = time.time() + timeout
    if fn is None:
        rep.line("    (point_reached() нет — жду %.0f с по таймеру)" % timeout)
        time.sleep(timeout)
        return
    while time.time() < deadline:
        try:
            if fn():
                return
        except Exception as e:
            rep.line("    point_reached() -> %s: %s, дальше жду по таймеру" % (type(e).__name__, e))
            time.sleep(max(0.0, deadline - time.time()))
            return
        time.sleep(0.05)
    rep.line("    ТАЙМАУТ ожидания точки (%.0f с)" % timeout)


def maneuver(pioneer, rep, label, question, x, y, z, yaw, settle):
    """Один манёвр: подсказка наблюдателю, команда, пауза, показания."""
    rep.line()
    rep.line("  --- %s ---" % label)
    rep.line("  СМОТРИТЕ: %s" % question)
    rep.line("  Команда: go_to_local_point(x=%.2f, y=%.2f, z=%.2f, yaw=%.1f)" % (x, y, z, yaw))
    try:
        pioneer.go_to_local_point(x, y, z, yaw)
    except Exception as e:
        rep.line("    ОШИБКА команды: %s: %s" % (type(e).__name__, e))
        return
    wait_point(pioneer, timeout=12.0, rep=rep)
    time.sleep(settle)
    pos, ori = read_state(pioneer)
    rep.line("    LPS position: %r" % (pos,))
    rep.line("    orientation:  %r" % (ori,))


def main():
    parser = argparse.ArgumentParser(description="Проверка осей и единиц yaw (полёт!)")
    parser.add_argument("--height", type=float, default=1.0, help="рабочая высота, м")
    parser.add_argument("--step", type=float, default=1.0, help="смещение в тестах, м")
    parser.add_argument("--settle", type=float, default=3.0, help="пауза на месте, с")
    parser.add_argument("--skip-yaw", action="store_true", help="не проверять разворот")
    args = parser.parse_args()

    try:
        from pioneer_sdk2 import Pioneer
    except Exception as e:
        print("Нет pioneer_sdk2: %s: %s" % (type(e).__name__, e))
        return 1

    rep = Report()
    rep.line("Проверка осей go_to_local_point и единиц yaw")
    rep.line("Высота %.2f м, шаг %.2f м" % (args.height, args.step))
    rep.line()
    rep.line("НАБЛЮДАТЕЛЬ: встаньте ПОЗАДИ дрона, лицом в ту же сторону, что и его нос.")
    rep.line("Тогда «вперёд» — от вас, «вправо» — направо от вас.")

    h, s = args.height, args.step
    pioneer = None
    try:
        pioneer = Pioneer()
        rep.line("\n  Подключение установлено")

        pos, ori = read_state(pioneer)
        rep.line("  До arm():  position=%r orientation=%r" % (pos, ori))

        rep.line("\n  arm() — моторы запускаются, отойдите. Ноль СК фиксируется здесь.")
        pioneer.arm()
        time.sleep(3.0)
        pos, ori = read_state(pioneer)
        rep.line("  После arm(): position=%r orientation=%r" % (pos, ori))

        pioneer.takeoff()
        time.sleep(3.0)
        pos, ori = read_state(pioneer)
        rep.line("  После takeoff(): position=%r orientation=%r" % (pos, ori))

        # Тест 1: первый аргумент. По раздатке организаторов это «вправо».
        maneuver(pioneer, rep, "ТЕСТ 1: первый аргумент (x)",
                 "дрон ушёл ВПЕРЁД, НАЗАД, ВПРАВО или ВЛЕВО?",
                 s, 0.0, h, 0.0, args.settle)
        maneuver(pioneer, rep, "возврат в центр", "дрон вернулся в исходную точку",
                 0.0, 0.0, h, 0.0, args.settle)

        # Тест 2: второй аргумент. По раздатке организаторов это «вперёд».
        maneuver(pioneer, rep, "ТЕСТ 2: второй аргумент (y)",
                 "дрон ушёл ВПЕРЁД, НАЗАД, ВПРАВО или ВЛЕВО?",
                 0.0, s, h, 0.0, args.settle)
        maneuver(pioneer, rep, "возврат в центр", "дрон вернулся в исходную точку",
                 0.0, 0.0, h, 0.0, args.settle)

        if not args.skip_yaw:
            # Тест 3: единицы yaw. 90 в градусах — четверть оборота; 90 в радианах —
            # это ~14 оборотов, дрон закрутится. Если увидели раскрутку — гасите полёт.
            rep.line()
            rep.line("  ВНИМАНИЕ: если после следующей команды дрон начнёт быстро крутиться,")
            rep.line("  это радианы — прерывайте Ctrl+C, скрипт посадит дрон.")
            maneuver(pioneer, rep, "ТЕСТ 3: единицы yaw",
                     "поворот примерно на ЧЕТВЕРТЬ оборота (градусы) или раскрутка (радианы)?",
                     0.0, 0.0, h, 90.0, args.settle)
            maneuver(pioneer, rep, "возврат курса", "дрон вернулся в исходный курс",
                     0.0, 0.0, h, 0.0, args.settle)

        rep.line("\n  Посадка")
        pioneer.land()
        time.sleep(3.0)

    except KeyboardInterrupt:
        rep.line("\n  Прервано оператором (Ctrl+C)")
    except Exception as e:
        rep.line("\n  ОШИБКА: %s: %s" % (type(e).__name__, e))
        import traceback
        for line in traceback.format_exc().splitlines():
            rep.line("    " + line)
    finally:
        if pioneer is not None:
            try:
                # wait_callback=False, чтобы аварийная посадка не заблокировалась
                # ожиданием события, которого уже не будет.
                pioneer.wait_callback = False
                try:
                    if not pioneer.is_landed():
                        pioneer.land()
                        time.sleep(2.0)
                except Exception:
                    pass
                pioneer.disarm()
                pioneer.close_connection()
                rep.line("  Дрон дизармлен, соединение закрыто")
            except Exception as e:
                rep.line("  Ошибка при закрытии: %s: %s" % (type(e).__name__, e))

    rep.line()
    rep.line("=" * 78)
    rep.line("КАК ЗАПОЛНИТЬ config.yaml -> axes")
    rep.line("=" * 78)
    rep.line("  Наша СК: X вперёд, Y влево, Z вверх (правая). Адаптер живёт в flight.py:")
    rep.line("      swap_xy: true   ->  sdk_x = sign_x * наш_y,  sdk_y = sign_y * наш_x")
    rep.line("      swap_xy: false  ->  sdk_x = sign_x * наш_x,  sdk_y = sign_y * наш_y")
    rep.line()
    rep.line("  Тест 1 показал, куда смотрит ось X автопилота, тест 2 — куда ось Y:")
    rep.line()
    rep.line("    тест 1     тест 2     swap_xy   sign_x   sign_y")
    rep.line("    ВПРАВО     ВПЕРЁД     true      -1       1      <- ожидание по раздатке оргов")
    rep.line("    ВЛЕВО      ВПЕРЁД     true       1       1")
    rep.line("    ВПЕРЁД     ВЛЕВО      false      1       1      <- как на сайте Geoscan")
    rep.line("    ВПЕРЁД     ВПРАВО     false      1      -1")
    rep.line("    НАЗАД      ВПРАВО     false     -1      -1")
    rep.line("    НАЗАД      ВЛЕВО      false     -1       1")
    rep.line("    ВПРАВО     НАЗАД      true      -1      -1")
    rep.line("    ВЛЕВО      НАЗАД      true       1      -1")
    rep.line()
    rep.line("  Проверить подстановкой: команда «1 м вперёд» (наш_x=1, наш_y=0) обязана")
    rep.line("  дать те sdk_x/sdk_y, которые в тестах увели дрон вперёд.")
    rep.line()
    rep.line("  Тест 3: четверть оборота -> yaw_units: deg; раскрутка -> yaw_units: rad.")
    rep.line("  Поворот в обратную сторону от ожидаемого -> yaw_sign: -1")

    rep.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), REPORT_NAME))
    return 0


if __name__ == "__main__":
    sys.exit(main())
