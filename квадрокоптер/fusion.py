# -*- coding: utf-8 -*-
"""Реестр станций: наблюдения -> подтверждённые станции -> пакеты наружу.

Здесь решается главное условие подзадачи 2.3.2: «количество обнаруженных станций
соответствует реальному». Оно даёт ВСЕ 8 баллов и устроено симметрично — лишняя
станция стоит ровно столько же, сколько пропущенная. Отсюда две меры:

  ассоциация по мировым координатам  одна станция, увиденная с двадцати точек змейки,
                                     обязана остаться одной записью, а не двадцатью;
  подтверждение несколькими кадрами  одиночная ложная детекция не должна становиться
                                     станцией.

Координата станции — МЕДИАНА по накопленным наблюдениям, а не среднее: медиана
не сдвигается от одного выброса, а выбросы у нас будут (docs/DRONE_PLAN.md §2.2).

Пакет station_new уходит СРАЗУ В ПОЛЁТЕ, как только станция подтвердилась, а не пачкой
при посадке: визуализатор оценивается «в режиме реального времени».

Модуль на стандартной библиотеке: ни cv2, ни numpy, ни SDK. Проверяется тестами целиком.
"""
from __future__ import annotations

import math
import threading
from statistics import median

# Те же три строки, что в protocol/messages.py -> STATUSES и в метках классов модели.
СТАТУСЫ = ("ok", "dust", "broken")

# Статус по умолчанию. Классический детектор классов не различает, но поле status
# в station_new обязательное по протоколу, и «исправна» — самое безопасное умолчание:
# автомобиль поедет собирать энергию, а не объедет станцию как сломанную.
СТАТУС_ПО_УМОЛЧАНИЮ = "ok"


class Station(object):
    """Одна станция в реестре: где она, сколько раз видена и чем считается.

    Наблюдения делятся на ТОЧНЫЕ и ГРУБЫЕ, и это не украшение. По бюджету ошибки
    (docs/DRONE_PLAN.md §2.2) в оценку координат идут только наблюдения из центральных
    50 % кадра. Но если отбрасывать остальные совсем, станция, прошедшая лишь по краю
    кадра, не попадёт в реестр ВООБЩЕ — а это минус станция в счёте, то есть все
    8 баллов подзадачи. Поэтому грубые наблюдения считаются наравне (станция найдена
    и подтверждена), а координату дают только точные; грубые — лишь пока точных нет.
    """

    def __init__(self, station_id, x, y, status=None, conf=1.0, precise=True,
                 max_obs=60):
        self.id = station_id
        self.max_obs = int(max_obs)
        self._точные = []               # [(x, y)] из центральной части кадра
        self._грубые = []               # [(x, y)] от краёв: хуже, но лучше, чем ничего
        self.status_votes = {}          # статус -> сумма уверенностей
        self.status = None
        self.conf = 0.0
        self.sent = False               # ушёл ли station_new
        self.status_sent = None         # какой статус ушёл последним
        self.sent_xy = None             # какие координаты ушли последними
        self.add(x, y, status, conf, precise)

    @property
    def n_obs(self):
        return len(self._точные) + len(self._грубые)

    @property
    def n_precise(self):
        return len(self._точные)

    def _точки(self):
        return self._точные or self._грубые

    @property
    def x(self):
        return median([x for x, _ in self._точки()])

    @property
    def y(self):
        return median([y for _, y in self._точки()])

    @property
    def precise(self):
        """Есть ли хоть одно наблюдение из центра кадра. Если нет, координата
        держится на грубых, и её нельзя предъявлять как «<30 см»."""
        return bool(self._точные)

    def distance_to(self, x, y):
        return math.hypot(self.x - x, self.y - y)

    def add(self, x, y, status=None, conf=1.0, precise=True):
        куда = self._точные if precise else self._грубые
        куда.append((float(x), float(y)))
        # Окно наблюдений ограничено: за длинный полёт позиция дрейфует, и старые
        # наблюдения тянут медиану к тому, где дрон думал, что он находится, полминуты
        # назад. Плюс это потолок памяти на борту.
        if len(куда) > self.max_obs:
            del куда[0]
        if status in СТАТУСЫ:
            # Голосуем взвешенно по уверенности: уверенная детекция должна перевешивать
            # сомнительную, а не просто добавлять голос.
            self.status_votes[status] = self.status_votes.get(status, 0.0) + float(conf)
        self._пересчитать_статус()

    def _пересчитать_статус(self):
        if not self.status_votes:
            self.status = None
            self.conf = 0.0
            return
        всего = sum(self.status_votes.values())
        лидер = max(self.status_votes, key=lambda s: self.status_votes[s])
        self.status = лидер
        self.conf = self.status_votes[лидер] / всего if всего else 0.0

    def reported_status(self):
        """Что писать в пакет. Без голосов — умолчание, иначе лидер голосования."""
        return self.status or СТАТУС_ПО_УМОЛЧАНИЮ

    def reported_conf(self):
        """Уверенность для пакета. Без голосов честнее отдать 0.0, чем выдумать 1.0:
        по этому числу потребитель решает, доверять ли статусу."""
        return round(self.conf, 3) if self.status else 0.0

    def as_dict(self):
        return {"id": self.id, "x": round(self.x, 3), "y": round(self.y, 3),
                "status": self.reported_status(), "conf": self.reported_conf(),
                "n_obs": self.n_obs, "n_precise": self.n_precise,
                "precise": self.precise, "sent": self.sent}

    def __repr__(self):
        return ("Station(%s, x=%.2f, y=%.2f, набл.=%d/%d, %s)"
                % (self.id, self.x, self.y, self.n_precise, self.n_obs,
                   self.reported_status()))


class StationRegistry(object):
    """Реестр станций дрона. Один на полёт.

    Живёт в потоке перцепции, а число станций читает поток миссии (для recon_done),
    поэтому всё под одним замком.
    """

    def __init__(self, cfg, transport=None, factory=None, state=None, task=None,
                 log=None, prefix=None):
        раздел = cfg.get("fusion", {})
        self.assoc_radius = float(раздел.get("assoc_radius", 0.40))
        # Физический минимум между центрами двух РАЗНЫХ станций. Всё, что ближе, —
        # это одна станция, как бы ни разошлись наблюдения. См. _ближайшая().
        self.min_separation = float(раздел.get("min_separation", 0.0))
        self.confirm_obs = int(раздел.get("confirm_obs", 3))
        self.max_obs = int(раздел.get("max_obs", 60))
        self.status_min_conf = float(раздел.get("status_min_conf", 0.6))
        self.max_stations = int(раздел.get("max_stations", 12))
        self.resend_moved = float(раздел.get("resend_moved", 0.10))

        self.transport = transport
        self.factory = factory
        self.state = state
        self.task = task
        self._log = log or (lambda text: print(text))
        # Префикс по источнику: у дрона d1, d2…, у автомобиля c1… Независимые нумерации
        # не должны столкнуться, а слияние «оба увидели одну станцию» делает потребитель
        # по координатам, а не по id (docs/DRONE_PLAN.md §2.3).
        self.prefix = prefix or str(cfg.get("mission", {}).get("src", "drone"))[0]

        self._lock = threading.Lock()
        self._stations = []
        self._next_id = 1
        self.observations = 0
        self.rejected_far = 0           # наблюдения, отброшенные как «слишком далеко»
        self.prevented_splits = 0       # сколько раз не дали одной станции стать двумя

    # ------------------------------------------------------------------ наблюдения

    def observe(self, x, y, status=None, conf=1.0, precise=True):
        """Одно наблюдение объекта в мировых координатах.

        precise=False — станция видна у края кадра: такое наблюдение подтверждает
        её существование, но координату не уточняет (см. Station).

        Возвращает Station, к которой оно отнеслось (новую или существующую).
        Отправку пакетов делает сама: держать её в вызывающем коде значит рано или
        поздно забыть про неё в одной из веток.
        """
        if x is None or y is None:
            return None
        if not (_конечное(x) and _конечное(y)):
            # Луч почти вдоль горизонта даёт координаты в километрах. Такое наблюдение
            # обязано быть отброшено, а не занесено в реестр.
            self.rejected_far += 1
            return None

        with self._lock:
            self.observations += 1
            станция = self._ближайшая(x, y)
            if станция is not None:
                станция.add(x, y, status, conf, precise)
            else:
                # Между assoc_radius и min_separation — МЁРТВАЯ ЗОНА: наблюдение
                # слишком далеко, чтобы уточнять координату, но ближе, чем физически
                # может стоять вторая станция (расстановку мерили рулеткой). Значит
                # это разбросанное наблюдение той же станции — чаще всего обрывок
                # панели, который детектор увидел отдельным пятном.
                # Заводить по нему новую запись нельзя: одна станция превратится
                # в две, а это те же 8 баллов, что и пропущенная.
                сосед = (self._ближайшая(x, y, радиус=self.min_separation)
                         if self.min_separation > self.assoc_radius else None)
                if сосед is not None:
                    self.prevented_splits += 1
                    # Наблюдение соседа, но ГРУБОЕ: подтверждать станцию оно вправе,
                    # а тянуть её координату к обрывку — нет.
                    сосед.add(x, y, status, conf, precise=False)
                    станция = сосед
                elif len(self._stations) >= self.max_stations:
                    # Больше станций, чем бывает на полигоне (регламент: 3-5), —
                    # признак того, что детектор сыпет ложными. Молча плодить записи
                    # нельзя: счёт станций и есть критерий 8 баллов.
                    self.rejected_far += 1
                    return None
                else:
                    станция = Station("%s%d" % (self.prefix, self._next_id), x, y,
                                      status, conf, precise, max_obs=self.max_obs)
                    self._next_id += 1
                    self._stations.append(станция)

            новая = self._подтвердилась(станция)
            уточнилась = self._координата_уехала(станция)
            смена = self._статус_изменился(станция)

        # Отправка — вне замка: сеть не должна держать поток перцепции.
        if новая or уточнилась:
            self._отправить_станцию(станция, повтор=not новая)
        elif смена:
            self._отправить_статус(станция)
        self._обновить_состояние()
        return станция

    def _ближайшая(self, x, y, радиус=None):
        """Станция ближе радиуса или None. По умолчанию радиус — assoc_radius:
        больше нашей точности, но меньше реального расстояния между станциями."""
        лучшая, расстояние = None, (self.assoc_radius if радиус is None else радиус)
        for станция in self._stations:
            d = станция.distance_to(x, y)
            if d < расстояние:
                лучшая, расстояние = станция, d
        return лучшая

    def _подтвердилась(self, станция):
        return not станция.sent and станция.n_obs >= self.confirm_obs

    def _координата_уехала(self, станция):
        """Стоит ли переслать station_new с уточнённой координатой.

        Пакет уходит после трёх наблюдений — и это верно, визуализатор оценивается
        в реальном времени. Но медиана трёх наблюдений заметно хуже медианы двадцати,
        а оценивают нас по итоговой точке на карте. Хаб и визуализатор ведут станции
        ПО id, поэтому повторный station_new с тем же id уточняет запись, а не плодит
        новую (см. hub.py -> HubState._apply).
        """
        if not станция.sent or self.resend_moved <= 0 or станция.sent_xy is None:
            return False
        return станция.distance_to(*станция.sent_xy) > self.resend_moved

    def _статус_изменился(self, станция):
        if not станция.sent or станция.status is None:
            return False
        if станция.status == станция.status_sent:
            return False
        # Уверенность ниже порога означает, что голоса поделились почти поровну.
        # Слать такое — значит мигать статусом на карте визуализатора.
        return станция.conf >= self.status_min_conf

    # -------------------------------------------------------------------- отправка

    def _отправить_станцию(self, станция, повтор=False):
        станция.sent = True
        станция.status_sent = станция.reported_status()
        станция.sent_xy = (станция.x, станция.y)
        self._log("[реестр] станция %s%s: x=%+.2f y=%+.2f статус=%s (набл. %d, из них "
                  "точных %d)%s"
                  % (станция.id, " уточнена" if повтор else "",
                     станция.x, станция.y, станция.reported_status(),
                     станция.n_obs, станция.n_precise,
                     "" if станция.precise else "  ВНИМАНИЕ: только по краю кадра"))
        if not self._можно("station_new"):
            return
        self._послать(self.factory.station_new(
            station_id=станция.id, x=станция.x, y=станция.y,
            status=станция.reported_status(), conf=станция.reported_conf()))

    def _отправить_статус(self, станция):
        предыдущий, станция.status_sent = станция.status_sent, станция.reported_status()
        self._log("[реестр] станция %s: статус %s -> %s (уверенность %.2f)"
                  % (станция.id, предыдущий, станция.status_sent, станция.conf))
        if not self._можно("status_update"):
            return
        self._послать(self.factory.status_update(
            station_id=станция.id, status=станция.reported_status(),
            conf=станция.reported_conf()))

    def _можно(self, что):
        if self.transport is None or self.factory is None:
            return False
        return self.task is None or self.task.can(что)

    def _послать(self, пакет):
        try:
            self.transport.send(пакет)
        except Exception as e:
            # Транспорт не имеет права ронять полёт: пакет потерян, полёт продолжается.
            self._log("[реестр] пакет не ушёл: %s: %s" % (type(e).__name__, e))

    def _обновить_состояние(self):
        """Число подтверждённых станций видно потоку миссии — из него берётся count
        для recon_done и строка HUD."""
        if self.state is not None:
            self.state.update(stations=self.confirmed_count())

    def flush(self):
        """Дослать уточнённые координаты всех станций. Зовётся один раз, в конце
        облёта, перед recon_done.

        Момент самый выгодный: все наблюдения собраны, медиана максимально устойчива,
        а автомобиль ещё не начал строить маршрут. Возвращает, сколько пакетов ушло.
        """
        with self._lock:
            надо = [с for с in self._stations
                    if с.sent and (с.sent_xy is None
                                   or с.distance_to(*с.sent_xy) > 1e-6)]
        for станция in надо:
            self._отправить_станцию(станция, повтор=True)
        return len(надо)

    # --------------------------------------------------------------------- выдача

    def confirmed_count(self):
        with self._lock:
            return sum(1 for с in self._stations if с.sent)

    def stations(self, confirmed_only=True):
        with self._lock:
            return [с for с in self._stations if с.sent or not confirmed_only]

    def snapshot(self):
        with self._lock:
            return [с.as_dict() for с in self._stations if с.sent]

    def summary(self):
        подтверждено = self.confirmed_count()
        with self._lock:
            всего = len(self._stations)
        текст = ("[реестр] наблюдений %d, станций подтверждено %d (кандидатов %d), "
                 "отброшено %d" % (self.observations, подтверждено,
                                   всего - подтверждено, self.rejected_far))
        if self.prevented_splits:
            # Не украшение: большое число означает, что детектор регулярно рвёт
            # панель на куски, и защита в реестре работает вместо лечения причины.
            текст += ("\n[реестр] предотвращено расколов станции на две: %d "
                      "(если много — поднять detector.close_m)"
                      % self.prevented_splits)
        return текст


def _конечное(значение):
    """Не NaN и не бесконечность, и вообще в пределах разумного полигона."""
    try:
        значение = float(значение)
    except (TypeError, ValueError):
        return False
    return значение == значение and abs(значение) < 1e4
