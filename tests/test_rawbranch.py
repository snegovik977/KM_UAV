# -*- coding: utf-8 -*-
"""Декодер сырых веток и проверка экспорта согласованы между собой.

Это самая дорогая ошибка всего проекта: raw-branch модель с плоским декодером даёт
мусор, плоская модель с нашим декодером — пустоту, и оба случая проходят БЕЗ ЕДИНОЙ
ОШИБКИ (docs/YOLO_TRAINING.md §6). Заметить это можно только на площадке, потеряв
попытку, поэтому форма выходов проверяется здесь — без torch, дрона и модели.

Формы взяты из настоящего экспорта 01.08.2026 (YOLO11n, форк airockchip, imgsz 320):
три масштаба по три ветки — box-DFL [1,64,H,W], классы [1,nc,H,W], score-sum [1,1,H,W].
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from perception.detector import decode_rawbranch                   # noqa: E402
import export_onnx                                                 # noqa: E402


def сырые_ветки(nc=2, img_size=320, заполнить=None):
    """Девять тензоров, как их отдаёт raw-branch экспорт."""
    выходы = []
    for stride in (8, 16, 32):
        g = img_size // stride
        box = np.zeros((1, 64, g, g), dtype="float32")
        cls = np.zeros((1, nc, g, g), dtype="float32")
        sum_ = np.zeros((1, 1, g, g), dtype="float32")
        if заполнить is not None:
            заполнить(stride, box, cls)
        выходы += [box, cls, sum_]
    return выходы


def test_ветки_опознаются_по_числу_каналов_а_не_по_порядку():
    """RKNN переставляет выходы, и полагаться на порядок нельзя — это записано
    в decode_rawbranch и должно оставаться правдой."""
    def один_объект(stride, box, cls):
        if stride == 8:
            cls[0, 1, 5, 5] = 0.9          # класс 1 в клетке (5,5)
            box[0, :, 5, 5] = 1.0          # равномерный DFL -> непустой бокс

    прямые = сырые_ветки(заполнить=один_объект)
    боксы, классы, скоры = decode_rawbranch(прямые, 2, 320, 0.5, 0.45)
    assert боксы is not None and len(боксы) == 1
    assert int(классы[0]) == 1
    assert скоры[0] == pytest.approx(0.9, abs=1e-5)

    # Тот же набор в обратном порядке обязан дать тот же результат.
    обратные = list(reversed(прямые))
    б2, к2, с2 = decode_rawbranch(обратные, 2, 320, 0.5, 0.45)
    assert б2 is not None and len(б2) == 1
    assert int(к2[0]) == int(классы[0])
    assert np.allclose(б2[0], боксы[0])


def test_бокс_попадает_в_кадр_и_центрируется_на_клетке():
    def один_объект(stride, box, cls):
        if stride == 8:
            cls[0, 0, 10, 20] = 0.8
            box[0, :, 10, 20] = 1.0

    боксы, _, _ = decode_rawbranch(сырые_ветки(заполнить=один_объект), 2, 320, 0.5, 0.45)
    x1, y1, x2, y2 = боксы[0]
    assert 0 <= x1 < x2 <= 320 and 0 <= y1 < y2 <= 320
    # Клетка (col=20, row=10) при stride 8 -> центр (164, 84) в пикселях входа.
    assert (x1 + x2) / 2 == pytest.approx(164.0, abs=1.0)
    assert (y1 + y2) / 2 == pytest.approx(84.0, abs=1.0)


def test_пустой_выход_это_ноль_детекций_а_не_падение():
    боксы, классы, скоры = decode_rawbranch(сырые_ветки(), 2, 320, 0.25, 0.45)
    assert боксы is None and классы is None and скоры is None


def test_число_классов_меняет_разбор():
    """Классовая ветка опознаётся по числу каналов = числу классов. Если модель
    обучена на другое их число, ветка просто не найдётся — детекций не будет."""
    боксы, _, _ = decode_rawbranch(сырые_ветки(nc=2), 3, 320, 0.5, 0.45)
    assert боксы is None


def test_проверка_экспорта_отвергает_плоскую_голову(tmp_path, capsys):
    """export_onnx.проверить_onnx обязана отличать раw-branch от плоской головы:
    именно плоская после INT8 даёт ноль детекций на борту."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto

    def файл_с_выходами(выходы):
        граф = helper.make_graph(
            [helper.make_node("Identity", ["вход"], [и])
             for и, _ in выходы],
            "тест",
            [helper.make_tensor_value_info("вход", TensorProto.FLOAT, [1, 3, 320, 320])],
            [helper.make_tensor_value_info(и, TensorProto.FLOAT, ф) for и, ф in выходы])
        путь = str(tmp_path / ("%d.onnx" % len(выходы)))
        onnx.save(helper.make_model(граф), путь)
        return путь

    плоская = файл_с_выходами([("out", [1, 6, 2100])])
    assert export_onnx.проверить_onnx(плоская, 2, log=lambda *_: None) is False

    сырая = []
    for i, g in enumerate((40, 20, 10)):
        сырая += [("box%d" % i, [1, 64, g, g]),
                  ("cls%d" % i, [1, 2, g, g]),
                  ("sum%d" % i, [1, 1, g, g])]
    assert export_onnx.проверить_onnx(файл_с_выходами(сырая), 2,
                                      log=lambda *_: None) is True
    # То же число веток, но модель на 3 класса — при nc=2 разбор невозможен.
    assert export_onnx.проверить_onnx(файл_с_выходами(сырая), 3,
                                      log=lambda *_: None) is False
