# -*- coding: utf-8 -*-
"""Центрирование по посадочному знаку «H» перед посадкой.

Две половины, как и у детекции станций. Первая — детектор знака находит жёлтый круг
на кадре заглушки, нарисованном той же дырочной моделью, которой локализация считает
координаты; вторая — центр знака проецируется обратно на землю с точностью в
сантиметры, и полётный автомат сводит дрон над ним.

Дрона и полигона тесты не требуют: заглушка камеры рисует знак со смещением от нуля
(имитируя увод локальной СК за облёт), а кинематика заглушки реально «летит» к цели,
поэтому итоговая точка посадки проверяется без единого взлёта.
"""
from __future__ import annotations

import threading
import time

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

import config as config_module
from flight import Flight
from localization import Pose, pixel_to_ground
from mission import LAND, RETURN, SURVEY, Mission, MissionState
from mock_pioneer import MockCamera, MockPioneer
from perception.calib import from_config as калибровка_из_конфига
from perception.landing import LandingGuide, LandingPadDetector, from_config
from tasks import TaskProfile


@pytest.fixture
def конфиг():
    return config_module.load()


@pytest.fixture
def калибровка(конфиг):
    return калибровка_из_конфига(конфиг, log=lambda _: None)


def детектор_из(конфиг):
    раздел = конфиг["landing"]
    return LandingPadDetector(
        hue_lo=раздел["hue_lo"], hue_hi=раздел["hue_hi"], sat_min=раздел["sat_min"],
        val_min=раздел["val_min"], close_px=раздел["close_px"],
        min_area_px=раздел["min_area_px"], min_fill=раздел["min_fill"],
        log=lambda _: None)


def кадр_со_знаком(pad, поза, конфиг):
    """Кадр заглушки с посадочным знаком в точке pad при заданной позе дрона."""
    камера = MockCamera(fps=1e6, stations=[], cfg=конфиг, pad=pad)
    камера.pose = lambda: поза
    return камера._синтетика()


# ------------------------------------------------------------------- детектор знака

def test_детектор_находит_знак_под_дроном(конфиг):
    поза = Pose(0.0, 0.0, 1.5, 0.0, 0.0, 0.0)
    frame = кадр_со_знаком((0.0, 0.0), поза, конфиг)

    детекция = детектор_из(конфиг).detect(frame)
    assert детекция is not None, "жёлтый знак прямо под дроном не найден"
    # Знак под дроном рисуется около центра кадра (серва не строго надир — небольшой снос).
    assert abs(детекция.cx - конфиг["camera"]["frame_width"] / 2.0) < 200
    assert детекция.score >= конфиг["landing"]["min_fill"]


def test_пустой_пол_не_даёт_ложного_знака(конфиг):
    поза = Pose(0.0, 0.0, 1.5, 0.0, 0.0, 0.0)
    frame = кадр_со_знаком(None, поза, конфиг)     # знак не рисуется
    assert детектор_из(конфиг).detect(frame) is None


def test_станции_не_принимаются_за_знак(конфиг):
    """Тёмные панели станций жёлтыми не бывают — детектор знака их игнорирует."""
    поза = Pose(1.2, 0.0, 1.5, 0.0, 0.0, 0.0)
    камера = MockCamera(fps=1e6, stations=[(1.2, 0.0, "ok"), (1.2, 0.6, "dust")],
                        cfg=конфиг, pad=None)
    камера.pose = lambda: поза
    assert детектор_из(конфиг).detect(камера._синтетика()) is None


# ------------------------------------------------ центр знака -> точка на земле

def test_центр_знака_локализуется_обратно(конфиг, калибровка):
    """Ключевое: пиксель центра знака обязан вернуться в его наземную точку с
    точностью в сантиметры — иначе центрирование сведёт дрон не туда."""
    pad = (0.30, -0.20)
    поза = Pose(0.0, 0.0, 1.5, 0.0, 0.0, 0.0)
    frame = кадр_со_знаком(pad, поза, конфиг)

    детекция = детектор_из(конфиг).detect(frame)
    assert детекция is not None
    точка = pixel_to_ground(детекция.cx, детекция.cy, поза,
                            калибровка.intrinsics, калибровка.r_mount)
    assert точка is not None
    ошибка = ((точка[0] - pad[0]) ** 2 + (точка[1] - pad[1]) ** 2) ** 0.5
    assert ошибка < 0.05, "центр знака локализован с ошибкой %.3f м" % ошибка


# ------------------------------------------------------------------------ гид

def test_гид_пишет_точку_знака_в_состояние(конфиг, калибровка):
    state = MissionState()
    state.update(state=LAND, position=(0.0, 0.0, 1.5))
    гид = LandingGuide(детектор_из(конфиг), калибровка, state,
                       phases=(RETURN, LAND), log=lambda _: None)

    frame = кадр_со_знаком((0.25, 0.15), Pose(0.0, 0.0, 1.5, 0.0, 0.0, 0.0), конфиг)
    гид.update(frame)

    s = state.snapshot()
    assert s["pad"] is not None, "гид не записал точку знака в состояние"
    ошибка = ((s["pad"][0] - 0.25) ** 2 + (s["pad"][1] - 0.15) ** 2) ** 0.5
    assert ошибка < 0.05


def test_гид_молчит_вне_фаз_посадки(конфиг, калибровка):
    """Во время облёта знак искать не надо: он появится в кадре у площадки и только
    зашумил бы состояние. Обновление вне RETURN/LAND ничего не пишет."""
    state = MissionState()
    state.update(state=SURVEY, position=(0.0, 0.0, 1.5))
    гид = LandingGuide(детектор_из(конфиг), калибровка, state,
                       phases=(RETURN, LAND), log=lambda _: None)

    гид.update(кадр_со_знаком((0.0, 0.0), Pose(0.0, 0.0, 1.5, 0.0, 0.0, 0.0), конфиг))
    assert state.snapshot()["pad"] is None


def test_гид_из_конфига_выключается_флагом(конфиг):
    state = MissionState()
    конфиг["landing"]["enabled"] = False
    assert from_config(конфиг, state, log=lambda _: None) is None


# --------------------------------------------------- сведение над знаком (сквозной)

def настройки(конфиг):
    """Ускоренные тайминги: тест не должен идти минуту."""
    конфиг["flight"]["arm_settle"] = 0.05
    конфиг["flight"]["takeoff_settle"] = 0.05
    конфиг["flight"]["settle"] = 0.02
    конфиг["flight"]["point_timeout"] = 3.0
    конфиг["mission"]["manual_start"] = True
    конфиг["mission"]["telemetry_hz"] = 20.0
    конфиг["loop"]["length"] = 2.0
    конфиг["loop"]["width"] = 2.0
    return конфиг


def test_дрон_сводится_над_смещённым_знаком(конфиг, калибровка):
    """Знак смещён от нуля на 30 см — как если бы оптопоток увёл локальную СК за облёт.
    Возврат приводит дрон в (0, 0), центрирование обязано доложить его на знак."""
    настройки(конфиг)
    pad = (0.30, -0.25)

    pioneer = MockPioneer(speed=6.0)          # станет _последним_дроном для камеры
    state = MissionState()
    flight = Flight(pioneer, конфиг, log=lambda _: None)
    mission = Mission(flight, None, конфиг, state=state, log=lambda _: None,
                      task=TaskProfile("2.3.1"))

    камера = MockCamera(fps=60.0, stations=[], cfg=конфиг, pad=pad)
    гид = LandingGuide(детектор_из(конфиг), калибровка, state,
                       phases=(RETURN, LAND), log=lambda _: None)

    # Поток перцепции: как в main.py, читает кадры и кладёт точку знака в состояние.
    стоп = threading.Event()

    def перцепция():
        while not стоп.is_set():
            гид.update(камера._синтетика())
            time.sleep(0.01)

    поток_перцепции = threading.Thread(target=перцепция, daemon=True)
    поток_перцепции.start()
    try:
        mission.run()
    finally:
        стоп.set()
        поток_перцепции.join(timeout=2.0)

    assert state.snapshot()["abort_reason"] is None
    x, y, _ = flight.position()
    ошибка = ((x - pad[0]) ** 2 + (y - pad[1]) ** 2) ** 0.5
    assert ошибка < конфиг["landing"]["tol"] + 0.05, (
        "дрон сел в (%.2f, %.2f), знак в (%.2f, %.2f), промах %.2f м"
        % (x, y, pad[0], pad[1], ошибка))
    assert pioneer.state == "ON_LAND"


def test_без_знака_садится_вслепую_в_ноль(конфиг):
    """Знак не виден (камеры в этом тесте нет) — центрирование пропускается, дрон
    садится по координатам возврата. Ветка обязана быть безопасной для 2.3.1."""
    настройки(конфиг)
    pioneer = MockPioneer(speed=6.0)
    state = MissionState()
    flight = Flight(pioneer, конфиг, log=lambda _: None)
    mission = Mission(flight, None, конфиг, state=state, log=lambda _: None,
                      task=TaskProfile("2.3.1"))
    mission.run()

    assert state.snapshot()["abort_reason"] is None
    x, y, _ = flight.position()
    assert abs(x) < 0.1 and abs(y) < 0.1, "без знака должен сесть в точку возврата"
    assert pioneer.state == "ON_LAND"
