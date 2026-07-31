#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Программа-передатчик: две роли, обе нужны до того, как заработает железо.

    takeoff  — играет за автомобиль: отдаёт дрону команду «на взлёт» и показывает
               его телеметрию. Так проверяется подзадача 2.3.1, пока трек автомобиля
               не готов.
    recon    — играет за дрона: шлёт три пакета регламента 2.2.4 (station_new,
               status_update, recon_done) с интервалом 5 секунд. Это заглушка,
               под которую трек автомобиля пишет свой планировщик маршрута,
               а визуализатор — отрисовку.

Обе роли ходят через один и тот же protocol/ — тот же, что и настоящий дрон.
В этом весь смысл: в финале дрон подставляется вместо передатчика, и код автомобиля
не должен заметить разницы (docs/NOTES.md §3).

Запуск:
    python transmitter.py takeoff --hub http://127.0.0.1:5001
    python transmitter.py recon   --hub http://127.0.0.1:5001 --stations 5
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocol import HttpTransport, MessageFactory, STATUSES  # noqa: E402

DEFAULT_HUB = "http://127.0.0.1:5001"
REGLAMENT_INTERVAL = 5.0     # «интервал пакетов от передатчика — 5 секунд» (2.2.4)


def role_takeoff(args):
    """Команда на взлёт + наблюдение за телеметрией дрона."""
    factory = MessageFactory("transmitter")
    transport = HttpTransport(args.hub, src="transmitter").start()

    health = transport.health()
    if not health.get("ok"):
        print("[передатчик] хаб не отвечает: %s" % health.get("error"))
        print("[передатчик] пакеты всё равно уйдут в очередь — хаб можно поднять позже")
    else:
        print("[передатчик] хаб жив: %r" % health)

    msg = factory.takeoff(origin_x=args.origin_x, origin_y=args.origin_y,
                          origin_yaw=args.origin_yaw,
                          zone_w=args.zone_w, zone_h=args.zone_h)
    transport.send(msg)
    print("[передатчик] отправлена команда на взлёт: %r" % msg["data"])
    print("[передатчик] жду телеметрию дрона, Ctrl+C — выход")

    last_state = None
    repeat_at = time.time() + args.repeat if args.repeat else None
    try:
        while True:
            for incoming in transport.recv():
                if incoming["type"] == "telemetry":
                    data = incoming["data"]
                    if data["state"] != last_state:
                        print("[дрон] состояние: %s" % data["state"])
                        last_state = data["state"]
                    print("\r[дрон] x=%+.2f y=%+.2f z=%+.2f yaw=%+.1f batt=%.2f  "
                          % (data["x"], data["y"], data["z"], data["yaw"], data["batt"]),
                          end="")
                    sys.stdout.flush()
                elif incoming["type"] == "recon_done":
                    print("\n[дрон] разведка завершена, станций: %d"
                          % incoming["data"]["count"])
                elif incoming["type"] == "station_new":
                    d = incoming["data"]
                    print("\n[дрон] станция %s: x=%+.2f y=%+.2f статус=%s conf=%.2f"
                          % (d["id"], d["x"], d["y"], d["status"], d["conf"]))
                elif incoming["type"] == "status_update":
                    d = incoming["data"]
                    print("\n[дрон] станция %s -> статус %s (conf=%.2f)"
                          % (d["id"], d["status"], d["conf"]))

            # Повтор команды: дрон мог ещё не подняться к моменту первой отправки.
            if repeat_at is not None and time.time() >= repeat_at and last_state is None:
                transport.send(factory.takeoff(origin_x=args.origin_x, origin_y=args.origin_y,
                                               origin_yaw=args.origin_yaw,
                                               zone_w=args.zone_w, zone_h=args.zone_h))
                print("\n[передатчик] дрон молчит — повтор команды на взлёт")
                repeat_at = time.time() + args.repeat
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[передатчик] остановлен")
    finally:
        print("[передатчик] %s" % transport.summary())
        transport.stop()


def _fake_stations(count, zone_w, zone_h, seed):
    """Правдоподобная расстановка: сетка с разбросом, ровно одна запылённая.

    Сетка, а не полный рандом: реальные станции организаторы расставляют по полигону
    с разносом, и потребителю (планировщику маршрута автомобиля) нужны данные,
    похожие на настоящие, а не станции в одной точке.
    """
    rng = random.Random(seed)
    cols = int(count ** 0.5 + 0.999) or 1
    rows = int((count + cols - 1) // cols)
    stations = []
    for i in range(count):
        col, row = i % cols, i // cols
        x = (col + 0.5) / cols * zone_w - zone_w / 2.0 + rng.uniform(-0.2, 0.2)
        y = (row + 0.5) / rows * zone_h - zone_h / 2.0 + rng.uniform(-0.2, 0.2)
        stations.append({"id": "d%d" % (i + 1), "x": round(x, 2), "y": round(y, 2),
                         "status": "ok", "conf": round(rng.uniform(0.72, 0.95), 2)})
    if stations:
        stations[rng.randrange(len(stations))]["status"] = "dust"
    if len(stations) > 2:
        rest = [s for s in stations if s["status"] == "ok"]
        rng.choice(rest)["status"] = "broken"
    return stations


def role_recon(args):
    """Эмуляция воздушной разведки: три пакета регламента 2.2.4."""
    factory = MessageFactory("transmitter")
    transport = HttpTransport(args.hub, src="transmitter").start()
    stations = _fake_stations(args.stations, args.zone_w, args.zone_h, args.seed)

    print("[передатчик] роль «дрон»: %d станций, интервал %.1f с"
          % (len(stations), args.interval))
    try:
        for station in stations:
            transport.send(factory.station_new(
                station_id=station["id"], x=station["x"], y=station["y"],
                status=station["status"], conf=station["conf"]))
            print("[передатчик] station_new %s: x=%+.2f y=%+.2f статус=%s"
                  % (station["id"], station["x"], station["y"], station["status"]))
            time.sleep(args.interval)

        # Уточнение статуса по мере накопления наблюдений — то же, что сделает
        # настоящий дрон, когда сменится лидер голосования (docs/DRONE_PLAN.md §3.1).
        if args.status_updates and stations:
            target = stations[0]
            new_status = next(s for s in STATUSES if s != target["status"])
            transport.send(factory.status_update(
                station_id=target["id"], status=new_status, conf=0.88))
            print("[передатчик] status_update %s -> %s" % (target["id"], new_status))
            time.sleep(args.interval)

        transport.send(factory.recon_done(count=len(stations)))
        print("[передатчик] recon_done: %d станций" % len(stations))

        # Даём потоку отправки дошептать очередь до конца.
        time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[передатчик] остановлен")
    finally:
        print("[передатчик] %s" % transport.summary())
        transport.stop()


def main():
    parser = argparse.ArgumentParser(description="Программа-передатчик (эмулятор)")
    parser.add_argument("--hub", default=DEFAULT_HUB, help="адрес хаба")
    sub = parser.add_subparsers(dest="role")

    p_takeoff = sub.add_parser("takeoff", help="за автомобиль: команда на взлёт")
    p_takeoff.add_argument("--origin-x", type=float, default=0.0, dest="origin_x")
    p_takeoff.add_argument("--origin-y", type=float, default=0.0, dest="origin_y")
    p_takeoff.add_argument("--origin-yaw", type=float, default=0.0, dest="origin_yaw")
    # По умолчанию размеры не шлём: габариты трассы меряет команда дрона, и присланное
    # наугад число не должно менять его маршрут.
    p_takeoff.add_argument("--zone-w", type=float, default=None, dest="zone_w")
    p_takeoff.add_argument("--zone-h", type=float, default=None, dest="zone_h")
    p_takeoff.add_argument("--repeat", type=float, default=5.0,
                           help="повторять команду каждые N с, пока дрон молчит (0 — не повторять)")
    p_takeoff.set_defaults(func=role_takeoff)

    p_recon = sub.add_parser("recon", help="за дрона: три пакета разведки")
    p_recon.add_argument("--stations", type=int, default=5)
    p_recon.add_argument("--zone-w", type=float, default=6.0, dest="zone_w")
    p_recon.add_argument("--zone-h", type=float, default=6.0, dest="zone_h")
    p_recon.add_argument("--interval", type=float, default=REGLAMENT_INTERVAL)
    p_recon.add_argument("--seed", type=int, default=1)
    p_recon.add_argument("--no-status-updates", action="store_false", dest="status_updates")
    p_recon.set_defaults(func=role_recon)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
