# -*- coding: utf-8 -*-
"""Реестр станций: ассоциация, подтверждение, медиана, голосование за статус.

Здесь проверяется условие ВСЕХ 8 баллов подзадачи 2.3.2 — «количество обнаруженных
станций соответствует реальному». Оно симметрично: и лишняя станция, и пропущенная
стоят одинаково, поэтому тесты идут парами «не разделил одну на две» и «не склеил две
в одну».

Ниже есть и сквозной прогон: заглушка камеры рисует станции той же дырочной моделью,
которой локализация считает координаты, поэтому «мок -> детектор -> локализация ->
реестр -> пакет» проверяется целиком, а критерий «<30 см» — числом, а не на глаз.
"""
from __future__ import annotations

import math

import pytest

import config as config_module
from fusion import Station, StationRegistry
from mission import MissionState
from protocol import MessageFactory
from tasks import TaskProfile


class ЗаписнойТранспорт(object):
    def __init__(self, ронять=False):
        self.отправленные = []
        self.ронять = ронять

    def send(self, msg):
        if self.ронять:
            raise IOError("хаб недоступен")
        self.отправленные.append(msg)


@pytest.fixture
def конфиг():
    return config_module.load()


def реестр(конфиг, task="2.3.2", transport=None, state=None, **правки):
    конфиг["fusion"].update(правки)
    return StationRegistry(конфиг, transport=transport,
                           factory=MessageFactory("drone") if transport else None,
                           state=state, task=TaskProfile(task), log=lambda _: None)


def пакеты(transport, тип):
    return [m for m in transport.отправленные if m["type"] == тип]


# ------------------------------------------------------------------- ассоциация

def test_одна_станция_с_многих_кадров_остаётся_одной(конфиг):
    """Станция видна с двадцати точек змейки. Двадцать записей вместо одной —
    и «количество обнаруженных» не сойдётся с реальным."""
    р = реестр(конфиг)
    for шаг in range(20):
        р.observe(1.0 + 0.02 * math.sin(шаг), -0.5 + 0.02 * math.cos(шаг))
    assert р.confirmed_count() == 1


def test_две_станции_не_склеиваются(конфиг):
    """Порог ассоциации обязан быть меньше реального расстояния между станциями."""
    р = реестр(конфиг)
    for _ in range(5):
        р.observe(0.0, 0.0)
        р.observe(1.5, 0.0)
    assert р.confirmed_count() == 2


def test_порог_ассоциации_соблюдается(конфиг):
    р = реестр(конфиг, assoc_radius=0.40)
    р.observe(0.0, 0.0)
    р.observe(0.39, 0.0)               # внутри радиуса — та же станция
    assert len(р.stations(confirmed_only=False)) == 1
    р.observe(0.0, 0.45)               # снаружи — новая
    assert len(р.stations(confirmed_only=False)) == 2


def test_ассоциация_идёт_к_ближайшей(конфиг):
    """Два кандидата в радиусе — наблюдение обязано уйти к ближнему, иначе
    соседние станции медленно сползаются в одну."""
    р = реестр(конфиг, assoc_radius=1.0)
    р.observe(0.0, 0.0)
    р.observe(1.5, 0.0)                # дальше радиуса — заводится вторая станция
    станции = р.stations(confirmed_only=False)
    assert len(станции) == 2
    р.observe(0.9, 0.0)                # до первой 0.9, до второй 0.6 — обе в радиусе
    assert станции[1].n_obs == 2 and станции[0].n_obs == 1


# ----------------------------------------------------------------- подтверждение

def test_одиночная_детекция_не_становится_станцией(конфиг):
    """Прямая защита от ложного срабатывания: одна вспышка не должна попасть
    ни в счёт станций, ни на карту визуализатора."""
    transport = ЗаписнойТранспорт()
    р = реестр(конфиг, transport=transport, confirm_obs=3)
    р.observe(1.0, 1.0)
    assert р.confirmed_count() == 0
    assert пакеты(transport, "station_new") == []


def test_station_new_уходит_ровно_на_третьем_наблюдении(конфиг):
    transport = ЗаписнойТранспорт()
    р = реестр(конфиг, transport=transport, confirm_obs=3)
    р.observe(1.0, 1.0)
    р.observe(1.02, 0.99)
    assert пакеты(transport, "station_new") == []
    р.observe(0.99, 1.01)
    assert len(пакеты(transport, "station_new")) == 1


def test_станция_не_раздваивается_при_повторах(конфиг):
    """Сколько бы пакетов о станции ни ушло, идентификатор у них один: хаб
    и визуализатор ведут станции по id, и лишней записи не появляется."""
    transport = ЗаписнойТранспорт()
    р = реестр(конфиг, transport=transport)
    for _ in range(30):
        р.observe(1.0, 1.0)
    отправленные = пакеты(transport, "station_new")
    assert len({m["data"]["id"] for m in отправленные}) == 1
    assert р.confirmed_count() == 1


def test_неподвижная_станция_шлётся_один_раз(конфиг):
    """Медиана не двигается — переотправлять нечего, эфир занимать незачем."""
    transport = ЗаписнойТранспорт()
    р = реестр(конфиг, transport=transport, resend_moved=0.10)
    for _ in range(30):
        р.observe(1.0, 1.0)
    assert len(пакеты(transport, "station_new")) == 1


def test_уточнённая_координата_дошлётся(конфиг):
    """Первый пакет уходит после трёх наблюдений — так требует «реальное время».
    Но оценивают нас по ИТОГОВОЙ точке, и уточнённую медиану надо доставить."""
    transport = ЗаписнойТранспорт()
    р = реестр(конфиг, transport=transport, resend_moved=0.10, assoc_radius=2.0)
    for _ in range(3):
        р.observe(0.0, 0.0)
    первый = пакеты(transport, "station_new")[-1]
    assert первый["data"]["x"] == pytest.approx(0.0)

    for _ in range(10):
        р.observe(1.0, 0.0)            # медиана уползает к 1.0
    последний = пакеты(transport, "station_new")[-1]
    assert последний["data"]["x"] > 0.5
    assert последний["data"]["id"] == первый["data"]["id"]


def test_flush_дошлёт_итоговые_координаты(конфиг):
    """Зовётся в конце облёта: все наблюдения собраны, медиана максимально
    устойчива, а автомобиль ещё не начал строить маршрут."""
    transport = ЗаписнойТранспорт()
    р = реестр(конфиг, transport=transport, resend_moved=0.0, assoc_radius=2.0)
    for _ in range(3):
        р.observe(0.0, 0.0)
    for _ in range(4):
        р.observe(0.6, 0.0)
    assert len(пакеты(transport, "station_new")) == 1, "переотправка выключена"

    assert р.flush() == 1
    последний = пакеты(transport, "station_new")[-1]
    assert последний["data"]["x"] == pytest.approx(0.6)
    assert р.flush() == 0, "повторный flush не должен слать ничего"


def test_идентификаторы_с_префиксом_источника(конфиг):
    """Нумерации дрона и автомобиля не должны столкнуться: у дрона d1, d2…"""
    р = реестр(конфиг)
    for _ in range(3):
        р.observe(0.0, 0.0)
    for _ in range(3):
        р.observe(2.0, 0.0)
    assert [с.id for с in р.stations()] == ["d1", "d2"]


# ---------------------------------------------------------------------- медиана

def test_координата_это_медиана_а_не_среднее(конфиг):
    """Один выброс не должен утащить координату за порог 30 см. Среднее — утащило бы."""
    р = реестр(конфиг, assoc_radius=5.0)
    for _ in range(10):
        р.observe(1.00, 2.00)
    р.observe(4.00, 2.00)              # грубый выброс
    станция = р.stations()[0]
    assert станция.x == pytest.approx(1.0, abs=1e-9)


def test_медиана_шумных_наблюдений_укладывается_в_30_см(конфиг):
    import random

    случайное = random.Random(20260731)
    р = реестр(конфиг, assoc_radius=1.0)
    истина = (1.3, -0.7)
    for _ in range(20):
        р.observe(истина[0] + случайное.gauss(0, 0.12),
                  истина[1] + случайное.gauss(0, 0.12))
    станция = р.stations()[0]
    ошибка = math.hypot(станция.x - истина[0], станция.y - истина[1])
    assert ошибка < 0.30


def test_окно_наблюдений_ограничено(конфиг):
    """За длинный полёт позиция дрейфует: старые точки обязаны выпадать, иначе
    медиана держится за то, где дрон думал, что он находится, полминуты назад."""
    р = реестр(конфиг, max_obs=10, assoc_radius=5.0)
    for _ in range(50):
        р.observe(0.0, 0.0)
    assert р.stations()[0].n_obs == 10


def test_дрейф_отслеживается_за_счёт_окна(конфиг):
    р = реестр(конфиг, max_obs=5, assoc_radius=5.0)
    for i in range(20):
        р.observe(0.1 * i, 0.0)
    # Последние пять наблюдений — 1.5…1.9, их медиана 1.7
    assert р.stations()[0].x == pytest.approx(1.7, abs=1e-9)


# --------------------------------------------- точные и грубые наблюдения

def test_станция_видная_только_с_краю_не_теряется(конфиг):
    """Прямая ловушка: если отбрасывать наблюдения от краёв кадра совсем, станция,
    прошедшая только по краю, не попадёт в реестр вообще — минус станция в счёте,
    то есть все 8 баллов подзадачи."""
    transport = ЗаписнойТранспорт()
    р = реестр(конфиг, transport=transport)
    for _ in range(4):
        р.observe(1.0, 1.0, precise=False)
    assert р.confirmed_count() == 1
    станция = р.stations()[0]
    assert not станция.precise, "координата держится на грубых наблюдениях"


def test_точные_наблюдения_вытесняют_грубые_из_координаты(конфиг):
    """Как только появилось хоть одно наблюдение из центра кадра, координату
    считаем по нему: край кадра локализуется заметно хуже."""
    р = реестр(конфиг, assoc_radius=2.0)
    for _ in range(10):
        р.observe(1.0, 0.0, precise=False)
    р.observe(1.3, 0.0, precise=True)
    станция = р.stations()[0]
    assert станция.precise
    assert станция.x == pytest.approx(1.3)
    assert станция.n_obs == 11 and станция.n_precise == 1


def test_грубые_наблюдения_считаются_для_подтверждения(конфиг):
    """Существование станции они подтверждают наравне — уточняют только координату."""
    р = реестр(конфиг, confirm_obs=3)
    р.observe(0.0, 0.0, precise=False)
    р.observe(0.0, 0.0, precise=False)
    assert р.confirmed_count() == 0
    р.observe(0.0, 0.0, precise=True)
    assert р.confirmed_count() == 1


# ------------------------------------------------------------ голосование за статус

def test_статус_по_умолчанию_ok_без_голосов(конфиг):
    """Классический детектор классов не различает, а поле status в пакете
    обязательное. «Исправна» безопаснее: автомобиль поедет за энергией."""
    transport = ЗаписнойТранспорт()
    р = реестр(конфиг, transport=transport)
    for _ in range(3):
        р.observe(0.0, 0.0)
    пакет = пакеты(transport, "station_new")[0]
    assert пакет["data"]["status"] == "ok"
    assert пакет["data"]["conf"] == 0.0     # честный ноль, а не выдуманная единица


def test_голосование_взвешено_по_уверенности(конфиг):
    """Две сомнительные детекции не должны перевесить одну уверенную."""
    р = реестр(конфиг, assoc_radius=5.0)
    р.observe(0.0, 0.0, "dust", 0.2)
    р.observe(0.0, 0.0, "dust", 0.2)
    р.observe(0.0, 0.0, "ok", 0.9)
    assert р.stations()[0].status == "ok"


def test_status_update_при_смене_лидера(конфиг):
    transport = ЗаписнойТранспорт()
    р = реестр(конфиг, task="2.3.3", transport=transport, status_min_conf=0.5)
    for _ in range(3):
        р.observe(0.0, 0.0, "ok", 0.9)
    assert пакеты(transport, "station_new")[0]["data"]["status"] == "ok"

    for _ in range(10):
        р.observe(0.0, 0.0, "dust", 0.95)
    обновления = пакеты(transport, "status_update")
    assert обновления and обновления[-1]["data"]["status"] == "dust"


def test_status_update_не_шлётся_при_шатком_голосовании(конфиг):
    """Голоса поровну — статус на карте мигал бы туда-сюда."""
    transport = ЗаписнойТранспорт()
    р = реестр(конфиг, task="2.3.3", transport=transport, status_min_conf=0.9)
    for _ in range(3):
        р.observe(0.0, 0.0, "ok", 1.0)
    for _ in range(3):
        р.observe(0.0, 0.0, "dust", 1.0)
    assert пакеты(transport, "status_update") == []


def test_подзадача_232_не_шлёт_status_update(конфиг):
    """Профиль 2.3.2 статусами не оценивается, и лишний пакет там не нужен."""
    transport = ЗаписнойТранспорт()
    р = реестр(конфиг, task="2.3.2", transport=transport, status_min_conf=0.5)
    for _ in range(3):
        р.observe(0.0, 0.0, "ok", 0.9)
    for _ in range(10):
        р.observe(0.0, 0.0, "dust", 0.95)
    assert len(пакеты(transport, "station_new")) == 1
    assert пакеты(transport, "status_update") == []


def test_подзадача_234_не_шлёт_ничего(конфиг):
    """Удаление пыли карту роя не строит: детекция нужна только для наведения."""
    transport = ЗаписнойТранспорт()
    р = реестр(конфиг, task="2.3.4", transport=transport)
    for _ in range(5):
        р.observe(0.0, 0.0, "dust", 0.9)
    assert transport.отправленные == []
    assert р.confirmed_count() == 1, "реестр обязан работать, даже когда не шлёт"


# ------------------------------------------------------------- защита от мусора

def test_бесконечные_координаты_отбрасываются(конфиг):
    """Луч почти вдоль горизонта даёт координаты в километрах. Такое наблюдение
    обязано быть отброшено, а не занесено в реестр."""
    р = реестр(конфиг)
    for значение in (float("nan"), float("inf"), 1e9, None):
        assert р.observe(значение, 0.0) is None
    assert р.stations(confirmed_only=False) == []
    assert р.rejected_far >= 3


def test_потолок_числа_станций(конфиг):
    """Больше станций, чем бывает на полигоне, — признак сыплющего детектора."""
    р = реестр(конфиг, max_stations=5, assoc_radius=0.1)
    for i in range(20):
        for _ in range(3):
            р.observe(i * 1.0, 0.0)
    assert len(р.stations(confirmed_only=False)) == 5


def test_упавший_транспорт_не_роняет_реестр(конфиг):
    """Недоступный хаб не имеет права прервать полёт: пакет теряется, реестр живёт."""
    transport = ЗаписнойТранспорт(ронять=True)
    р = реестр(конфиг, transport=transport)
    for _ in range(5):
        р.observe(0.0, 0.0)
    assert р.confirmed_count() == 1


def test_состояние_миссии_видит_число_станций(конфиг):
    """Из него поток миссии берёт count для recon_done."""
    state = MissionState()
    р = реестр(конфиг, state=state)
    for _ in range(3):
        р.observe(0.0, 0.0)
    for _ in range(3):
        р.observe(2.0, 0.0)
    assert state.snapshot()["stations"] == 2


def test_станция_печатается_читаемо(конфиг):
    станция = Station("d1", 1.0, 2.0, "dust", 0.8)
    assert "d1" in repr(станция) and "dust" in repr(станция)


# --------------------------------------------------- сквозной прогон без дрона

def test_сквозной_прогон_мок_детектор_локализация_реестр(конфиг):
    """Главная проверка пункта 2 без полигона: расставили станции в заглушке,
    пролетели над ними, сверили пришедшие координаты с расстановкой.

    Чего этот тест НЕ доказывает: правильности интринсик и R_mount. Заглушка рисует
    теми же числами, которыми потом считает, поэтому любая калибровка выглядит
    идеальной. Настоящую проверку даёт только рулетка под дроном.
    """
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")

    from localization import Pose, ground_area, in_central_region, pixel_to_ground
    from mock_pioneer import MockCamera
    from perception.calib import from_config as калибровка_из_конфига
    from perception.detector import create_detector

    расстановка = [(1.10, 0.45, "ok"), (1.10, -0.45, "dust")]
    калибровка = калибровка_из_конфига(конфиг, log=lambda _: None)
    детектор = create_detector(конфиг, log=lambda _: None)
    камера = MockCamera(fps=100000.0, stations=расстановка, cfg=конфиг)

    transport = ЗаписнойТранспорт()
    р = реестр(конфиг, transport=transport)

    # Полоса змейки: дрон идёт вперёд на рабочей высоте, станции проезжают под ним.
    # Шаг 4.5 см — это скорость 0.45 м/с при инференсе 10 Гц, как в config.yaml.
    for шаг in range(60):
        поза = Pose(-0.4 + 0.045 * шаг, 0.0, 2.0, 0.0, 0.0, 0.0)
        камера.pose = lambda поза=поза: поза
        frame = камера._синтетика()

        def площадь(углы, поза=поза):
            return ground_area(углы, поза, калибровка.intrinsics, калибровка.r_mount)

        for детекция in детектор.detect(frame, area_m2=площадь):
            точка = pixel_to_ground(детекция.cx, детекция.cy, поза,
                                    калибровка.intrinsics, калибровка.r_mount)
            if точка is None:
                continue
            в_центре = in_central_region(детекция.cx, детекция.cy,
                                         калибровка.width, калибровка.height)
            р.observe(точка[0], точка[1], детекция.label, детекция.score,
                      precise=в_центре)

    р.flush()
    отправленные = пакеты(transport, "station_new")
    найденные = {m["data"]["id"]: m for m in отправленные}
    assert len(найденные) == len(расстановка), (
        "число станций не сошлось: %d вместо %d — это все 8 баллов подзадачи"
        % (len(найденные), len(расстановка)))

    for x_истина, y_истина, _ in расстановка:
        ближайший = min(найденные.values(),
                        key=lambda m: math.hypot(m["data"]["x"] - x_истина,
                                                 m["data"]["y"] - y_истина))
        ошибка = math.hypot(ближайший["data"]["x"] - x_истина,
                            ближайший["data"]["y"] - y_истина)
        assert ошибка < 0.30, ("станция (%.2f, %.2f) уехала на %.3f м — это 4 балла"
                               % (x_истина, y_истина, ошибка))

    assert all(с.precise for с in р.stations()), (
        "ни одно наблюдение не попало в центральные 50 % кадра: при таком шаге змейки "
        "координаты держатся на краях кадра, где они хуже всего")
