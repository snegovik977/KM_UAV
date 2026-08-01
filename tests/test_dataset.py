# -*- coding: utf-8 -*-
"""Конвейер обучения: разметка, сборка датасета, сверка классов с бортом.

Ни torch, ни ultralytics, ни дрона эти тесты не требуют — проверяется ровно то,
что ломается молча: негодная разметка, протёкший в val клип и расхождение списка
классов между training/classes.txt и detector.labels в config.yaml.
"""
from __future__ import annotations

import os

import pytest

import common
import build_dataset


def test_классов_не_меньше_двух():
    """При одном классе классовая ветка модели неотличима от score-sum."""
    имена = common.классы()
    assert len(имена) >= 2


def test_классы_совпадают_с_конфигом_борта():
    """classes.txt и detector.labels — один и тот же список, иначе декодер на борту
    разложит сырые ветки RKNN не по тем каналам."""
    метки = common.метки_из_конфига()
    if метки is None:
        pytest.skip("config.yaml борта не прочитался")
    assert метки == common.классы()


def test_негодная_разметка_отбраковывается():
    число = len(common.классы())
    assert common.проверить_бокс((0, 0.5, 0.5, 0.2, 0.3), число) is None
    assert common.проверить_бокс((число, 0.5, 0.5, 0.2, 0.3), число)      # класса нет
    assert common.проверить_бокс((0, 1.7, 0.5, 0.2, 0.3), число)          # центр вне кадра
    assert common.проверить_бокс((0, 0.5, 0.5, 0.0, 0.3), число)          # нулевая ширина


def test_разметка_переживает_запись_и_чтение(tmp_path):
    путь = str(tmp_path / "кадр.txt")
    боксы = [(0, 0.5, 0.5, 0.2, 0.3), (1, 0.25, 0.75, 0.1, 0.1)]
    common.писать_разметку(путь, боксы)
    прочитанное = common.читать_разметку(путь)
    assert len(прочитанное) == 2
    for исходный, обратно in zip(боксы, прочитанное):
        assert обратно[0] == исходный[0]
        assert обратно[1:] == pytest.approx(исходный[1:], abs=1e-5)


def test_пустой_файл_это_негатив_а_не_ошибка(tmp_path):
    путь = str(tmp_path / "фон.txt")
    common.писать_разметку(путь, [])
    assert common.читать_разметку(путь) == []
    assert os.path.exists(путь)          # файл есть: кадр проверен, станций нет


def _клип(корень, имя, кадры):
    """Клип из кадров вида (номер, боксы|None). None = кадр без .txt, не размечен."""
    import numpy as np

    папка = os.path.join(корень, имя)
    os.makedirs(папка)
    for номер, боксы in кадры:
        картинка = np.zeros((16, 16, 3), dtype="uint8")
        common.imwrite(os.path.join(папка, "%s_%03d.jpg" % (имя, номер)), картинка)
        if боксы is not None:
            common.писать_разметку(
                os.path.join(папка, "%s_%03d.txt" % (имя, номер)), боксы)


def test_в_датасет_идут_только_размеченные_кадры(tmp_path):
    """Кадр без .txt — это «ещё не смотрели», а не негатив: в обучении он учил бы
    сеть не видеть станцию там, где она есть."""
    cv2 = pytest.importorskip("cv2")                                     # noqa: F841
    разметка = str(tmp_path / "images_to_label")
    os.makedirs(разметка)
    _клип(разметка, "первый", [(1, [(0, 0.5, 0.5, 0.2, 0.2)]),
                               (2, []),                    # проверенный негатив
                               (3, None)])                 # не размечен
    _клип(разметка, "второй", [(1, [(1, 0.5, 0.5, 0.2, 0.2)])])

    куда = str(tmp_path / "dataset")
    статистика, брак, по_классам = build_dataset.собрать(
        разметка, куда, ["второй"], common.классы(), log=lambda *_: None)

    assert брак == []
    assert статистика["train"]["первый"]["кадров"] == 2       # третий не взят
    assert статистика["train"]["первый"]["пустых"] == 1
    assert статистика["val"]["второй"]["кадров"] == 1
    assert по_классам[0] == 1 and по_классам[1] == 1
    # Имя клипа попадает в имя файла: кадры разных клипов лежат одной кучей.
    файлы = os.listdir(os.path.join(куда, "images", "train"))
    assert all(имя.startswith(("первый__", "второй__")) for имя in файлы)


def test_кадры_одного_клипа_не_попадают_в_оба_раздела(tmp_path):
    """Главная ловушка метрики: соседние кадры пролёта почти идентичны, и клип,
    разъехавшийся по train и val, даёт красивый mAP на слепой модели."""
    pytest.importorskip("cv2")
    разметка = str(tmp_path / "images_to_label")
    os.makedirs(разметка)
    for имя in ("клипA", "клипB"):
        _клип(разметка, имя, [(i, [(0, 0.5, 0.5, 0.2, 0.2)]) for i in range(4)])

    куда = str(tmp_path / "dataset")
    build_dataset.собрать(разметка, куда, ["клипB"], common.классы(), log=lambda *_: None)

    в_train = {и.split("__")[0] for и in os.listdir(os.path.join(куда, "images", "train"))}
    в_val = {и.split("__")[0] for и in os.listdir(os.path.join(куда, "images", "val"))}
    assert в_train == {"клипA"}
    assert в_val == {"клипB"}
    assert not (в_train & в_val)


def test_негодный_бокс_бракует_весь_кадр(tmp_path):
    pytest.importorskip("cv2")
    разметка = str(tmp_path / "images_to_label")
    os.makedirs(разметка)
    _клип(разметка, "битый", [(1, [(0, 0.5, 0.5, 0.2, 0.2)]),
                              (2, [(0, 1.5, 0.5, 0.2, 0.2)])])   # центр вне кадра

    куда = str(tmp_path / "dataset")
    статистика, брак, _ = build_dataset.собрать(
        разметка, куда, [], common.классы(), log=lambda *_: None)
    assert len(брак) == 1
    assert статистика["train"]["битый"]["кадров"] == 1


def test_data_yaml_содержит_классы_по_порядку(tmp_path):
    куда = str(tmp_path / "dataset")
    os.makedirs(куда)
    имена = common.классы()
    путь = build_dataset.написать_data_yaml(куда, имена)
    with open(путь, "rb") as f:
        текст = f.read().decode("utf-8")
    assert "nc: %d" % len(имена) in текст
    for i, имя in enumerate(имена):
        assert "  %d: %s" % (i, имя) in текст
    # Путь абсолютный: ultralytics ищет датасет не относительно data.yaml.
    assert os.path.abspath(куда) in текст


def test_val_выбирается_детерминированно():
    """Автовыбор val не смотрит на содержимое и потому не может «подобрать» удобный."""
    клипы = ["a", "b", "c", "d", "e", "f"]
    первый = build_dataset.выбрать_val(клипы, [], log=lambda *_: None)
    второй = build_dataset.выбрать_val(клипы, [], log=lambda *_: None)
    assert первый == второй
    assert первый and set(первый) <= set(клипы)
    # Заданный руками список побеждает автовыбор.
    assert build_dataset.выбрать_val(клипы, ["c"], log=lambda *_: None) == ["c"]
