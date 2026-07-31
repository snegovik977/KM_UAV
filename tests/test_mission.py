# -*- coding: utf-8 -*-
"""Автомат миссии на заглушке SDK: штатный проход и все аварийные ветки.

Аварийные переходы проверяются здесь, а не в воздухе, по двум причинам: в воздухе
разряженная батарея и срыв навигации воспроизводятся дорого и опасно, а по регламенту
1.3 время утекает, даже когда полигон простаивает.
"""
from __future__ import annotations

import threading
import time

import pytest

import config as config_module
from flight import Flight
from mission import AbortMission, Mission, MissionState
from mock_pioneer import MockPioneer


def настройки(зона=2.0, правки=None):
    """Конфиг с ускоренными таймингами: тест не должен идти минуту.

    Маленькая трасса — самый быстрый прогон. Тестам аварийных веток нужен облёт
    подлиннее, они просят контур побольше.
    """
    cfg = config_module.load()
    cfg["flight"]["arm_settle"] = 0.05
    cfg["flight"]["takeoff_settle"] = 0.05
    cfg["flight"]["settle"] = 0.02
    cfg["flight"]["point_timeout"] = 3.0
    cfg["mission"]["manual_start"] = True
    cfg["mission"]["telemetry_hz"] = 20.0
    cfg["loop"]["length"] = зона
    cfg["loop"]["width"] = зона
    cfg["zone"]["width"] = зона
    cfg["zone"]["height"] = зона
    for путь, значение in (правки or {}).items():
        раздел, ключ = путь.split(".")
        cfg[раздел][ключ] = значение
    return cfg


def собрать(cfg, transport=None, **аргументы_дрона):
    pioneer = MockPioneer(speed=5.0, **аргументы_дрона)
    state = MissionState()
    flight = Flight(pioneer, cfg, log=lambda text: None)
    mission = Mission(flight, transport, cfg, state=state, log=lambda text: None)
    return pioneer, mission, state


# ------------------------------------------------------------------ штатный проход

def test_полный_проход_состояний():
    cfg = настройки()
    pioneer, mission, state = собрать(cfg)
    mission.run()

    assert state.snapshot()["abort_reason"] is None
    события = [запись for _, запись in pioneer.journal]
    # Порядок обязателен: автопилот игнорирует команды «из прошлого состояния»
    assert события[0] == "arm"
    assert события[1] == "takeoff"
    assert "land" in события and "disarm" in события
    assert события.index("land") < события.index("disarm")
    assert pioneer.state == "ON_LAND"


def test_вся_змейка_пройдена():
    cfg = настройки()
    pioneer, mission, state = собрать(cfg)
    mission.run()

    итог = state.snapshot()
    assert итог["waypoints_total"] > 0
    assert итог["waypoint"] == итог["waypoints_total"], "облёт оборвался на середине"
    перелёты = [запись for _, запись in pioneer.journal if запись.startswith("goto(")]
    # Подъём на рабочую высоту + все точки змейки + возврат
    assert len(перелёты) >= итог["waypoints_total"] + 2


def test_дрон_возвращается_в_точку_взлёта():
    """2 балла подзадачи 2.3.1 — приземлиться в пределах посадочной площадки."""
    cfg = настройки()
    pioneer, mission, state = собрать(cfg)
    mission.run()
    x, y, _ = pioneer.get_local_position_lps()
    assert abs(x) < 0.1 and abs(y) < 0.1, "сел не там, где взлетел: (%.2f, %.2f)" % (x, y)


def test_состояние_дрона_видно_снаружи_во_время_полёта():
    """Поток перцепции читает MissionState, пока миссия летит: HUD и запись
    не должны ждать окончания перелёта."""
    cfg = настройки()
    pioneer, mission, state = собрать(cfg)
    поток = threading.Thread(target=mission.run, daemon=True)
    поток.start()

    состояния = set()
    предел = time.time() + 20.0
    while поток.is_alive() and time.time() < предел:
        состояния.add(state.snapshot()["state"])
        time.sleep(0.01)
    поток.join(timeout=5.0)

    assert {"ARM", "TAKEOFF", "SURVEY", "LAND"} <= состояния, состояния


# --------------------------------------------------------------- аварийные ветки

def test_низкий_заряд_прерывает_облёт():
    """Возврат по низкому заряду обязан сработать в середине змейки, а не после неё."""
    cfg = настройки(зона=6.0)
    порог = cfg["flight"]["batt_low"]
    pioneer, mission, state = собрать(cfg, batt=порог + 0.05, drain=0.05)
    mission.run()

    итог = state.snapshot()
    assert итог["abort_reason"] is not None
    assert "заряд" in итог["abort_reason"]
    assert итог["waypoint"] < итог["waypoints_total"], "облёт почему-то дошёл до конца"
    assert pioneer.state == "ON_LAND", "дрон остался в воздухе после аварии"


def test_критический_заряд_сажает_на_месте():
    """При критическом заряде возвращаться уже нечем — садимся здесь."""
    cfg = настройки()
    pioneer, mission, state = собрать(cfg, batt=cfg["flight"]["batt_critical"] - 0.1,
                                      drain=0.0)
    mission.run()
    assert "критический" in state.snapshot()["abort_reason"]
    assert pioneer.state == "ON_LAND"


def test_зависший_перелёт_ловится_таймаутом():
    """point_reached() навсегда False: без таймаута дрон висел бы до разряда."""
    cfg = настройки(правки={"mission.survey_timeout": 2.0})
    pioneer, mission, state = собрать(cfg, stuck=True)
    начало = time.time()
    mission.run()
    прошло = time.time() - начало

    assert "таймаут" in state.snapshot()["abort_reason"]
    assert прошло < 30.0, "таймаут не сработал вовремя"
    assert pioneer.state == "ON_LAND"


def test_срыв_навигации_прерывает_миссию():
    """Оптопоток над однотонным полом срывается, и координаты уезжают. Лучше сесть,
    чем привезти карту станций с ошибкой в метры."""
    # Зона побольше: реакция наступает на устойчивой потере (2 с), и облёт должен
    # длиться дольше этого окна, иначе тест проверял бы не то.
    cfg = настройки(зона=6.0)
    pioneer, mission, state = собрать(cfg, nav_fail=0.3)
    mission.run()
    assert "навигаци" in (state.snapshot()["abort_reason"] or "")
    assert pioneer.state == "ON_LAND"


def test_одиночный_сбой_навигации_не_прерывает():
    """Реагируем на устойчивую потерю, а не на первое же False: иначе любая помеха
    сорвёт попытку."""
    cfg = настройки()
    pioneer, mission, state = собрать(cfg)

    вызовы = {"n": 0}
    исходный = pioneer.get_nav_status_lps

    def мигающий():
        вызовы["n"] += 1
        return вызовы["n"] % 7 != 0        # каждый седьмой опрос — сбой

    pioneer.get_nav_status_lps = мигающий
    mission.run()
    pioneer.get_nav_status_lps = исходный

    assert state.snapshot()["abort_reason"] is None
    assert вызовы["n"] > 10, "проверка навигации вообще не выполнялась"


def test_остановка_снаружи():
    cfg = настройки()
    pioneer, mission, state = собрать(cfg)
    поток = threading.Thread(target=mission.run, daemon=True)
    поток.start()
    time.sleep(0.5)
    mission.stop("оператор нажал стоп")
    поток.join(timeout=15.0)

    assert not поток.is_alive()
    assert state.snapshot()["abort_reason"] == "оператор нажал стоп"
    assert pioneer.state == "ON_LAND"


def test_ошибка_в_середине_миссии_сажает_дрон():
    """Любое неожиданное исключение обязано закончиться посадкой, а не зависанием."""
    cfg = настройки()
    pioneer, mission, state = собрать(cfg)

    def взрыв(*args, **kwargs):
        raise RuntimeError("сломалось")

    mission._survey = взрыв
    mission.run()

    assert state.snapshot()["abort_reason"].startswith("RuntimeError")
    assert pioneer.state == "ON_LAND"


# ------------------------------------------------------------------- взаимодействие

class ЗаписнойТранспорт(object):
    """Транспорт без сети: отдаёт заготовленные пакеты, копит отправленные."""

    def __init__(self, входящие=None):
        self.входящие = list(входящие or [])
        self.отправленные = []

    def recv(self):
        полученное, self.входящие = self.входящие, []
        return полученное

    def send(self, msg):
        self.отправленные.append(msg)


def test_миссия_ждёт_команду_на_взлёт():
    """Без пакета takeoff дрон не имеет права взлетать: подзадача 2.3.1 звучит
    как «квадрокоптер получает сигнал от передатчика, выполняет взлёт»."""
    from protocol import MessageFactory

    cfg = настройки()
    cfg["mission"]["manual_start"] = False
    transport = ЗаписнойТранспорт()
    pioneer, mission, state = собрать(cfg, transport=transport)
    mission.factory = MessageFactory("drone")

    поток = threading.Thread(target=mission.run, daemon=True)
    поток.start()
    time.sleep(0.4)
    assert state.snapshot()["state"] == "IDLE"
    assert not pioneer.armed, "взлетел без команды"

    transport.входящие.append(MessageFactory("transmitter").takeoff(
        origin_x=1.0, origin_y=2.0, origin_yaw=90.0, zone_w=2.0, zone_h=2.0))
    поток.join(timeout=30.0)

    assert not поток.is_alive()
    assert state.snapshot()["origin"] == (1.0, 2.0, 90.0)
    assert pioneer.state == "ON_LAND"


def test_телеметрия_уходит_в_полёте():
    from protocol import MessageFactory

    cfg = настройки()
    transport = ЗаписнойТранспорт()
    pioneer, mission, state = собрать(cfg, transport=transport)
    mission.factory = MessageFactory("drone")
    mission.run()

    телеметрия = [m for m in transport.отправленные if m["type"] == "telemetry"]
    assert len(телеметрия) > 5
    состояния = {m["data"]["state"] for m in телеметрия}
    assert "SURVEY" in состояния


def test_origin_сдвигает_координаты_в_телеметрии():
    """В финале начало общей СК задаёт автомобиль. Дрон обязан отдавать координаты
    в ней, иначе карта визуализатора уедет относительно карты автомобиля."""
    from protocol import MessageFactory

    cfg = настройки()
    transport = ЗаписнойТранспорт()
    pioneer, mission, state = собрать(cfg, transport=transport)
    mission.factory = MessageFactory("drone")
    state.update(origin=(10.0, -5.0, 90.0))

    # Точка (1, 0) в СК дрона при повороте общей СК на 90°: наш X (вперёд)
    # становится осью Y общей СК, плюс сдвиг начала.
    assert mission._to_world(1.0, 0.0) == pytest.approx((10.0, -4.0))
    assert mission._to_world(0.0, 1.0) == pytest.approx((9.0, -5.0))
    # Нулевой origin ничего не меняет — обычный случай подзадач 2.3.x
    state.update(origin=(0.0, 0.0, 0.0))
    assert mission._to_world(1.5, -1.0) == pytest.approx((1.5, -1.0))

    state.update(origin=(10.0, -5.0, 90.0))
    mission.run()
    телеметрия = [m for m in transport.отправленные if m["type"] == "telemetry"]
    assert телеметрия, "телеметрия не ушла"
    # Координаты в пакетах — общие, а не собственные координаты дрона
    assert all(m["data"]["y"] > -8.0 for m in телеметрия)
    assert any(abs(m["data"]["x"] - 10.0) > 0.1 for m in телеметрия), \
        "поворот и сдвиг origin не применены"


def test_габариты_из_пакета_takeoff_по_флагу():
    """По умолчанию трассу меряем мы, и пакет её габариты не перекрывает. Но если
    договоримся, что размеры присылает автомобиль, это включается одним флагом."""
    from protocol import MessageFactory

    cfg = настройки()
    cfg["mission"]["manual_start"] = False
    cfg["loop"]["length"] = 6.0
    cfg["loop"]["width"] = 6.0
    cfg["loop"]["use_packet_zone"] = True
    transport = ЗаписнойТранспорт([MessageFactory("transmitter").takeoff(
        zone_w=1.5, zone_h=1.5)])
    pioneer, mission, state = собрать(cfg, transport=transport)
    mission.factory = MessageFactory("drone")
    mission.run()

    ширина = mission.plan.zone[1] - mission.plan.zone[0]
    assert ширина == pytest.approx(1.5), "габариты из пакета проигнорированы"


def test_пакет_не_меняет_трассу_по_умолчанию():
    """Обратный случай: чужая программа прислала размеры наугад — маршрут не поехал."""
    from protocol import MessageFactory

    cfg = настройки()
    cfg["mission"]["manual_start"] = False
    cfg["loop"]["length"] = 2.0
    cfg["loop"]["width"] = 2.0
    transport = ЗаписнойТранспорт([MessageFactory("transmitter").takeoff(
        zone_w=9.0, zone_h=9.0)])
    pioneer, mission, state = собрать(cfg, transport=transport)
    mission.factory = MessageFactory("drone")
    mission.run()

    ширина = mission.plan.zone[1] - mission.plan.zone[0]
    assert ширина == pytest.approx(2.0)


def test_recon_done_не_уходит_без_станций():
    """Пункт 1 станций ещё не ищет. Пустой recon_done заставил бы автомобиль
    строить маршрут по пустой карте."""
    from protocol import MessageFactory

    cfg = настройки()
    transport = ЗаписнойТранспорт()
    pioneer, mission, state = собрать(cfg, transport=transport)
    mission.factory = MessageFactory("drone")
    mission.run()
    assert not [m for m in transport.отправленные if m["type"] == "recon_done"]


def test_recon_done_уходит_после_облёта():
    """Пакет 3 регламента 2.2.4 обязан уйти строго после всей змейки."""
    from protocol import MessageFactory

    cfg = настройки()
    transport = ЗаписнойТранспорт()
    pioneer, mission, state = собрать(cfg, transport=transport)
    mission.factory = MessageFactory("drone")
    исходный = mission._survey

    def облёт_с_находками():
        # Реестр станций появится в пункте 2 плана; здесь подставляем его результат,
        # чтобы проверить сам порядок: recon_done строго после всей змейки.
        исходный()
        state.update(stations=3)
        mission._after_survey()

    mission._survey = облёт_с_находками
    mission.run()

    пакеты = [m for m in transport.отправленные if m["type"] == "recon_done"]
    assert len(пакеты) == 1
    assert пакеты[0]["data"]["count"] == 3


# ------------------------------------------------------------------------ прочее

def test_исключение_прерывания_несёт_причину():
    with pytest.raises(AbortMission) as поймано:
        raise AbortMission("низкий заряд", severity="land")
    assert поймано.value.severity == "land"
    assert "заряд" in поймано.value.reason


def test_hud_читаем():
    state = MissionState()
    state.update(state="SURVEY", position=(1.234, -2.5, 2.0), yaw=45.0,
                 battery=7.81, waypoint=3, waypoints_total=8, started=time.time())
    строка = state.hud()
    assert "SURVEY" in строка and "3/8" in строка and "7.81" in строка

