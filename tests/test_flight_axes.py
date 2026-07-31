# -*- coding: utf-8 -*-
"""Адаптер системы координат и защита по высоте.

Ошибка в осях — самая дорогая из «тихих»: она не падает с исключением, а разворачивает
всю змейку на 90°, и увидеть это можно только в воздухе, потратив попытку и лимит
времени. Поэтому преобразование проверяется на всех восьми возможных конфигурациях,
а не только на той, которую мы считаем верной.
"""
from __future__ import annotations

import itertools
import math

import pytest

import config as config_module
from flight import AxisAdapter, Flight, wrap_deg


ВСЕ_КОНФИГУРАЦИИ = list(itertools.product((True, False), (1, -1), (1, -1)))


@pytest.mark.parametrize("swap_xy, sign_x, sign_y", ВСЕ_КОНФИГУРАЦИИ)
def test_преобразование_обратимо(swap_xy, sign_x, sign_y):
    adapter = AxisAdapter(swap_xy=swap_xy, sign_x=sign_x, sign_y=sign_y)
    for x, y, z in [(0, 0, 0), (1, 0, 2), (0, 1, 2), (-3.5, 2.25, 1.5), (7.1, -0.4, 0.3)]:
        back = adapter.from_sdk(*adapter.to_sdk(x, y, z))
        assert back == pytest.approx((x, y, z)), "потеря при обратном преобразовании"


def test_вариант_из_раздатки_организаторов():
    """SDK: Y вперёд, X вправо. Наша СК: X вперёд, Y влево."""
    adapter = AxisAdapter(swap_xy=True, sign_x=-1, sign_y=1)
    # 1 метр вперёд по-нашему -> метр по оси Y автопилота, ноль по X
    assert adapter.to_sdk(1.0, 0.0, 2.0) == pytest.approx((0.0, 1.0, 2.0))
    # 1 метр влево по-нашему -> метр ВПРАВО-отрицательный по оси X автопилота
    assert adapter.to_sdk(0.0, 1.0, 2.0) == pytest.approx((-1.0, 0.0, 2.0))
    # Высота не трогается ни при какой конфигурации
    assert adapter.to_sdk(0.0, 0.0, 1.7)[2] == 1.7


def test_вариант_с_сайта_geoscan():
    adapter = AxisAdapter(swap_xy=False, sign_x=1, sign_y=1)
    assert adapter.to_sdk(1.0, 0.0, 2.0) == pytest.approx((1.0, 0.0, 2.0))
    assert adapter.to_sdk(0.0, 1.0, 2.0) == pytest.approx((0.0, 1.0, 2.0))


def test_радианы_и_градусы():
    в_градусах = AxisAdapter(yaw_units="deg")
    в_радианах = AxisAdapter(yaw_units="rad")
    assert в_градусах.yaw_to_sdk(90.0) == pytest.approx(90.0)
    assert в_радианах.yaw_to_sdk(90.0) == pytest.approx(math.pi / 2)
    for adapter in (в_градусах, в_радианах):
        assert adapter.yaw_from_sdk(adapter.yaw_to_sdk(37.0)) == pytest.approx(37.0)


def test_обратный_знак_yaw():
    adapter = AxisAdapter(yaw_sign=-1)
    assert adapter.yaw_to_sdk(90.0) == pytest.approx(-90.0)
    assert adapter.yaw_from_sdk(-90.0) == pytest.approx(90.0)


def test_негодные_параметры_не_принимаются():
    with pytest.raises(ValueError):
        AxisAdapter(sign_x=0)
    with pytest.raises(ValueError):
        AxisAdapter(yaw_units="градусы")


@pytest.mark.parametrize("угол, ожидание", [
    (0, 0), (180, 180), (-180, 180), (190, -170), (-190, 170), (540, 180), (359, -1),
])
def test_нормализация_угла(угол, ожидание):
    assert wrap_deg(угол) == pytest.approx(ожидание)


# ------------------------------------------------------------------ защита по высоте

class ЗаписнойPioneer:
    """Минимальная заглушка: запоминает, что именно ушло в автопилот."""

    def __init__(self, position=(0.0, 0.0, 1.0)):
        self.команды = []
        self._position = position

    def go_to_local_point(self, x, y, z, yaw, *args):
        self.команды.append((x, y, z, yaw))
        return True

    def go_to_local_point_body_fixed(self, x, y, z, yaw, *args):
        self.команды.append((x, y, z, yaw))
        return True

    def get_local_position_lps(self):
        # Ровно как на борту: без аргументов. Пример организаторов зовёт его
        # с позиционным True — на нашем SDK это TypeError.
        return self._position

    def point_reached(self):
        return True


@pytest.fixture
def настройки():
    return config_module.load()


def test_пол_по_высоте_срабатывает(настройки):
    """Ни одна команда ниже h_min не имеет права уйти в автопилот: замены сломанного
    оборудования по регламенту нет, а время попытки при поломке не останавливается."""
    pioneer = ЗаписнойPioneer()
    flight = Flight(pioneer, настройки, log=lambda text: None)
    flight.goto(1.0, 0.0, 0.05)
    assert flight.clamped == 1
    ушло_z = pioneer.команды[-1][2]
    assert ушло_z == pytest.approx(flight.h_min)


def test_потолок_по_высоте(настройки):
    pioneer = ЗаписнойPioneer()
    flight = Flight(pioneer, настройки, log=lambda text: None)
    flight.goto(0.0, 0.0, 10.0)
    assert pioneer.команды[-1][2] == pytest.approx(flight.h_max)


def test_относительный_спуск_тоже_под_полом(настройки):
    """goto_body сама по себе абсолютной высоты не знает — её обязан подставить Flight,
    иначе пошаговое снижение над панелью (подзадача 2.3.4) пробьёт пол."""
    pioneer = ЗаписнойPioneer(position=(0.0, 0.0, 0.5))
    flight = Flight(pioneer, настройки, log=lambda text: None)
    flight.goto_body(0.0, 0.0, -0.4)      # с 0.5 м вниз на 40 см = 0.1 м, ниже пола
    assert flight.clamped == 1
    # В СК автопилота Z не переставляется, поэтому dz читается прямо из команды.
    dz = pioneer.команды[-1][2]
    assert 0.5 + dz == pytest.approx(flight.h_min)


def test_позиция_возвращается_в_нашей_ск(настройки):
    """Позиция от автопилота обязана пройти обратное преобразование, иначе телеметрия
    и карта станций окажутся повёрнуты относительно того, куда дрон реально летал."""
    настройки["axes"]["swap_xy"] = True
    настройки["axes"]["sign_x"] = -1
    настройки["axes"]["sign_y"] = 1
    pioneer = ЗаписнойPioneer(position=(0.0, 1.0, 2.0))   # СК автопилота: метр вперёд
    flight = Flight(pioneer, настройки, log=lambda text: None)
    assert flight.position() == pytest.approx((1.0, 0.0, 2.0))   # наша СК: метр вперёд


def test_команда_и_позиция_согласованы(настройки):
    """Слетать в точку и прочитать позицию должно давать ту же точку — на всех
    конфигурациях осей сразу."""
    for swap_xy, sign_x, sign_y in ВСЕ_КОНФИГУРАЦИИ:
        настройки["axes"].update(swap_xy=swap_xy, sign_x=sign_x, sign_y=sign_y)
        pioneer = ЗаписнойPioneer()
        flight = Flight(pioneer, настройки, log=lambda text: None)
        flight.goto(2.0, -1.0, 1.5)
        sdk_x, sdk_y, sdk_z, _ = pioneer.команды[-1]
        pioneer._position = (sdk_x, sdk_y, sdk_z)
        assert flight.position() == pytest.approx((2.0, -1.0, 1.5))


# --------------------------------------------------------------- расхождения SDK

def test_дальномер_подбирается_автоматически(настройки):
    """Раздатка организаторов зовёт метод get_dist_sensor_data, сайт Geoscan —
    get_ranger_data с кортежем из пяти дальностей. Работать должно с любым."""

    class ТолькоRanger(ЗаписнойPioneer):
        def get_ranger_data(self):
            return (1.0, 1.0, 2.0, 2.0, 0.85)     # вертикальная — последняя

    class ТолькоDist(ЗаписнойPioneer):
        def get_dist_sensor_data(self):
            return 0.85

    for pioneer in (ТолькоRanger(), ТолькоDist()):
        flight = Flight(pioneer, настройки, log=lambda text: None)
        assert flight.range_down() == pytest.approx(0.85)

    # Нет ни того, ни другого — не исключение, а None: высота по дальномеру
    # это страховка, а не обязательный источник.
    assert Flight(ЗаписнойPioneer(), настройки, log=lambda text: None).range_down() is None


def test_позиция_с_позиционным_аргументом(настройки):
    """Пример организаторов зовёт get_local_position_lps(True)."""

    class ТребуетАргумент(ЗаписнойPioneer):
        def get_local_position_lps(self, update):
            assert update is True
            return (0.0, 0.0, 1.0)

    flight = Flight(ТребуетАргумент(), настройки, log=lambda text: None)
    assert flight.position() is not None


def test_на_земле_ли_без_метода_is_landed(настройки):
    """На нашем борту метода is_landed() у Pioneer НЕТ (check_sdk.py 31.07.2026),
    хотя он есть в примерах документации. Раньше прямой вызов бросал AttributeError,
    тот глушился, и дрон получал disarm в воздухе — то есть падал."""

    class СоСостояниемПолёта(ЗаписнойPioneer):
        def __init__(self, имя):
            ЗаписнойPioneer.__init__(self)
            self.имя = имя

        def get_fly_state(self):
            return type("FlyState", (), {"name": self.имя})()

    flight = Flight(СоСостояниемПолёта("LANDED"), настройки, log=lambda text: None)
    assert flight.is_landed() is True
    flight = Flight(СоСостояниемПолёта("IN_SKY"), настройки, log=lambda text: None)
    assert flight.is_landed() is False


def test_на_земле_ли_по_дальномеру(настройки):
    """Последний рубеж, когда нет ни is_landed(), ни get_fly_state()."""

    class ТолькоДальномер(ЗаписнойPioneer):
        def __init__(self, высота):
            ЗаписнойPioneer.__init__(self)
            self.высота = высота

        def get_dist_sensor_data(self):
            return self.высота

    assert Flight(ТолькоДальномер(0.02), настройки, log=lambda t: None).is_landed()
    assert not Flight(ТолькоДальномер(1.5), настройки, log=lambda t: None).is_landed()

    # Ничего не известно -> считаем, что дрон в воздухе. Лишняя команда land()
    # безобидна, пропущенная означает падение.
    assert not Flight(ЗаписнойPioneer(position=None), настройки,
                      log=lambda t: None).is_landed()


def test_заглушка_повторяет_отсутствие_is_landed():
    """Заглушка обязана врать так же, как железо: иначе тесты зелёные,
    а на борту AttributeError в аварийной посадке."""
    from mock_pioneer import MockPioneer

    pioneer = MockPioneer()
    try:
        assert not hasattr(pioneer, "is_landed"), \
            "у заглушки появился is_landed(), которого нет на борту"
        assert hasattr(pioneer, "get_fly_state")
        # Ranger на борту не установлен и отдаёт пять None — высота берётся
        # из get_dist_sensor_data.
        assert pioneer.get_ranger_data() == (None, None, None, None, None)
    finally:
        pioneer.close_connection()


def test_статус_навигации_читается_по_имени(настройки):
    """get_nav_status_lps отдаёт перечисление NavStatus, а не bool. Сравнивать
    по числу нельзя: значения могут поменяться между версиями SDK."""

    class СНавигацией(ЗаписнойPioneer):
        def __init__(self, имя):
            ЗаписнойPioneer.__init__(self)
            self.имя = имя

        def get_nav_status_lps(self):
            return type("NavStatus", (), {"name": self.имя, "__str__": lambda s: self.имя})()

    assert Flight(СНавигацией("OK"), настройки, log=lambda t: None).nav_ok()
    assert not Flight(СНавигацией("UNDEFINED"), настройки, log=lambda t: None).nav_ok()
    # Метода нет вовсе — не повод прерывать миссию
    assert Flight(ЗаписнойPioneer(), настройки, log=lambda t: None).nav_ok()


def test_отсутствие_телеметрии_не_роняет(настройки):
    """Автопилот может отдать None на любой запрос — это не повод падать в полёте."""

    class Молчун(ЗаписнойPioneer):
        def get_local_position_lps(self):
            return None

        def get_orientation(self):
            raise RuntimeError("нет связи")

        def get_battery_status(self):
            return None

    flight = Flight(Молчун(), настройки, log=lambda text: None)
    assert flight.position() is None
    assert flight.orientation() is None
    assert flight.battery() is None
    assert flight.nav_ok() is True      # неизвестно != авария


# ------------------------------------------------------------------------ конфиг

def test_конфиг_читается_без_pyyaml():
    """PyYAML нет ни здесь, ни, вероятно, на борту — разбор обязан работать своими силами."""
    cfg = config_module.loads(
        "hub:\n"
        "  url: \"http://10.0.0.1:5001\"   # комментарий с # внутри\n"
        "  timeout: 0.5\n"
        "axes:\n"
        "  swap_xy: true\n"
        "  sign_x: -1\n"
        "mission:\n"
        "  manual_start: false\n"
    )
    assert cfg["hub"]["url"] == "http://10.0.0.1:5001"
    assert cfg["hub"]["timeout"] == 0.5
    assert cfg["axes"]["swap_xy"] is True
    assert cfg["axes"]["sign_x"] == -1
    assert cfg["mission"]["manual_start"] is False


def test_боевой_конфиг_содержит_всё_нужное(настройки):
    for путь in ("hub.url", "axes.swap_xy", "axes.yaw_units", "camera.servo_angle",
                 "flight.h_min", "flight.h_max", "flight.h_survey", "flight.speed",
                 "zone.width", "zone.height", "zone.fov_h_deg", "zone.overlap",
                 "mission.telemetry_hz", "mission.survey_timeout"):
        настройки.get_path(путь)
    assert настройки.get_path("flight.h_min") < настройки.get_path("flight.h_survey")
    assert настройки.get_path("flight.h_survey") <= настройки.get_path("flight.h_max")


def test_список_в_конфиге_отвергается():
    """Разбор намеренно узкий: молча съеденный список хуже понятной ошибки."""
    with pytest.raises(config_module.ConfigError):
        config_module.loads("zone:\n  points:\n    - 1\n    - 2\n")
