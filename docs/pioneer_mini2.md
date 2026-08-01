# Geoscan Pioneer Mini 2 — выжимка из документации

Источник: https://docs.geoscan.ru/pioneer/instructions/pioneer-mini2/main-mini2.html
Собрано: 2026-07-31. Ниже — только то, что нужно для проекта CV на борту БПЛА.

> ⚠️ Раздатка организаторов расходится с этим документом по нескольким пунктам
> (порт `ImageViewer`, оси `go_to_local_point`, имя метода дальномера) —
> см. [organizer_handouts.md](organizer_handouts.md) §1 и §3.
>
> ⚠️ Полевой опыт другой команды на этом же дроне расходится с обоими источниками
> (диапазон `set_angle`, чип для RKNN, поведение `CameraType.OPT`) —
> см. [lessons_from_archipelago.md](lessons_from_archipelago.md) §1.

---

## 1. Что это за аппарат

Учебный квадрокоптер с бортовым ИИ-компьютером. Ключевое для нас: на борту стоит
ARM-микрокомпьютер (Radxa Zero 3W-класса) с 4 ГБ RAM и NPU, поэтому нейросети
крутятся **на самом дроне**, а не на ноутбуке.

| Параметр | Значение |
|---|---|
| Процессор | ARM, 4 ГБ RAM, NPU (RKNN) |
| Камера основная | 13 Мп, до 2K, 30 FPS |
| Камера оптического потока | диапазон 0.06–4 м |
| Wi-Fi | Wi-Fi 6, до 286.8 Мбит/с |
| Макс. горизонтальная скорость | 15 км/ч |
| Время полёта | до 11 мин |
| Масса | < 150 г |
| Батарея | Li-Po/Li-Ion, 3.85 В, 900 мА·ч, 45 г |
| Рабочая температура | 0…+40 °C |

Есть «функция доверенной среды»: логирование параметров полёта + geofence-контроль
выполнения миссии.

---

## 2. Подключение к дрону

Это самое важное — всё остальное делается через браузер.

1. Включить дрон (длинное нажатие кнопки питания).
2. Подключиться к Wi-Fi точке дрона:
   - **SSID**: `PMINI2-<UUID>` (у других плат — `RaZero-…`, `PiZero-…`)
   - **Пароль**: `geoscan123`
3. IP дрона: **`172.17.49.2`**

⚠️ Сайт Geoscan и опыт другой команды называют `10.42.0.1`. **Наш борт отвечает
на `172.17.49.2`** (проверено на площадке 31.07.2026) — этот адрес и прописан в
`launch.py`, `квадрокоптер/config.yaml` и [RUN.md](../RUN.md) §7. Ниже адреса
сервисов даны для нашего борта; если попадётся другой экземпляр, хост смотреть
как шлюз Wi-Fi-подключения (`ipconfig` / `ip route`).

### Порты / сервисы

| URL | Что это |
|---|---|
| `http://172.17.49.2:9090/` | **Pioneer Code** — главный хаб, отсюда всё остальное |
| `http://172.17.49.2:9999/` | **Code OSS** — VS Code в браузере, пишем и запускаем Python прямо на борту |
| `http://172.17.49.2:7777/` | **AI Models** — реестр моделей, загрузка `.rknn` |
| `http://172.17.49.2:8889/<stream-name>` | **Видеопоток** от `ImageViewer` (сюда смотрим результат CV). Порт **8889**, а не 8089 из документации — проверено 31.07.2026 |
| — | **Pioneer Bricks** — блочное программирование, нам не нужно |
| — | **Медиагалерея** — фото/видео с борта (только Mini 2) |

Ещё доступ: SSH и встроенный терминал CodeOSS. Настройка сети — утилита `sudo pionet`.

---

## 3. Pioneer OS

Linux-дистрибутив на базе Ubuntu. Предустановлено из коробки:

- **OpenCV**
- **NumPy**
- **pioneer_sdk2** — управление полётом + камера
- **pioneer_rknn** — инференс нейросетей на NPU

То есть `pip install` для основного стека не нужен. (Для Raspberry Pi Zero SDK2
пришлось бы ставить отдельно — к Mini 2 не относится.)

---

## 4. Pioneer SDK2 — управление полётом

```python
from pioneer_sdk2 import Pioneer

pioneer = Pioneer()
```

### Конструктор `Pioneer(...)`

| Параметр | Тип | По умолчанию | Смысл |
|---|---|---|---|
| `serial` | str | — | последовательный порт |
| `tcp` | str | `127.0.0.1:20556` | TCP-адрес автопилота |
| `baudrate` | int | `57600` | скорость порта |
| `wait_callback` | bool | `True` | блокирующие методы (ждать подтверждения) |
| `safety_command` | bool | `True` | проверка состояния перед командой |
| `logger` | bool | `True` | лог в консоль |

### Конечный автомат состояний

Строгая последовательность: **`ON_LAND → ARMED → IN_SKY → ON_LAND`**

При `wait_callback=True` + `safety_command=True`:
- команды «из прошлого состояния» игнорируются;
- пропуск обязательной стадии → `RuntimeError`;
- выполняются только команды текущего состояния.

То есть нельзя вызвать `takeoff()` без `arm()`.

### Полёт

```python
arm(timeout=5, retries=0)   # запуск моторов
disarm()                    # остановка моторов
takeoff()                   # взлёт
land()                      # посадка
rtl()                       # возврат домой
```

### Перемещение

```python
go_to_local_point(x, y, z, yaw, time=0)
go_to_local_point_body_fixed(x, y, z, yaw, time=0)   # в системе координат корпуса
go_to_global_point(latitude, longitude, altitude, yaw=0)
set_manual_speed(vx, vy, vz, yaw_rate, interval=1.0)
set_yaw(yaw)
```

`go_to_local_point_body_fixed` — то, что нужно для визуального следования за целью:
смещение относительно текущего положения дрона.

### Телеметрия

```python
get_battery_status()   # (voltage, temperature)
get_orientation()      # (roll, pitch, yaw)
get_accel()            # (x, y, z)
get_gyro()             # угловые скорости
get_altitude()         # высота по барометру
get_motors_rpm()       # обороты 4 моторов
```

### Навигация

```python
NavSystem.GPS   # глобальная
NavSystem.LPS   # локальная (УЗ/ИК)
NavSystem.OPT   # оптическая

get_nav_system(update=False)
get_nav_status_lps()
get_local_position_lps()      # (x, y, z)
get_global_position_gps()     # (lat, lon, alt)
get_satellites_count()
```

### События (неблокирующая альтернатива)

```python
from pioneer_sdk2 import Pioneer, Event

def callback(event):
    ...

pioneer.subscribe(callback, Event.TAKEOFF_COMPLETE)
pioneer.unsubscribe(callback, Event.TAKEOFF_COMPLETE)
```

Доступные события: `COPTER_LANDED`, `TAKEOFF_COMPLETE`, `ENGINES_STARTED`,
`POINT_REACHED`, `SHOCK` + события по батарее.

Для CV-цикла это правильный подход: основной поток крутит инференс, полётная
логика реагирует на события.

### Полезная нагрузка (Mini 2)

```python
grab_open(movement_time=0, velocity=100)
grab_close(movement_time=0, velocity=100)
grab_stop()
get_ranger_data()   # (right, left, forward, backward, vertical)
```

### RC-каналы

```python
send_rc_channels(channel_1=0, channel_2=0, channel_3=0, channel_4=0,
                 channel_5=1, channel_6=0, channel_7=1, channel_8=0)
```
Требует предварительной настройки параметров автопилота.

### Соединение и параметры

```python
pioneer.connect()            # вызывается автоматически в __init__
pioneer.close_connection()   # ОБЯЗАТЕЛЬНО в конце скрипта
pioneer.reboot_board()       # перезагрузка автопилота (Pi/Radxa)

pioneer.set_param("name", value)
pioneer.get_param("name", update=False)
```

Методы возвращают значение или `None` при ошибке; полётные команды — bool.

---

## 5. Камера

Драйверы: **gstreamer** (через shared memory) и **RTSP**.

```python
from pioneer_sdk2 import Camera, CameraType, ImageViewer

camera = Camera(camera_type=CameraType.MAIN)
frame = camera.get_cv_frame(timeout=5.0)   # numpy-массив (BGR), готов для OpenCV
camera.stop()

viewer = ImageViewer()
viewer.imshow(name="stream", frame=frame, fps=30)
viewer.close()
```

`ImageViewer` не открывает окно — он отдаёт поток в браузер на
`http://172.17.49.2:8889/<name>` — порт **8889**, проверено сканированием портов борта
31.07.2026 (на сайте Geoscan указан 8089, он закрыт). Это наш способ смотреть результат детекции.

> ⚠️ **Не открывать `CameraType.OPT`.** Это камера оптического потока, на которой держится
> позиционирование. Полевой опыт: второй потребитель мешает полёту и грузит CPU. Работаем
> только с `MAIN`, и только из одного процесса — камера отдаётся через shared memory
> (`/shmpipe`), при конфликте `get_cv_frame()` падает в `TimeoutError` («shm_open failed»).
> Всегда звать `camera.stop()` в `finally`.
> См. [lessons_from_archipelago.md](lessons_from_archipelago.md) §1.2.

### Поворотная камера (только Mini 2)

```python
from pioneer_sdk2 import ServoCamera, ServoPriority

servo_camera = ServoCamera()
servo_camera.set_angle(angle=-45, priority=ServoPriority.LOW)
```

> ⚠️ **Диапазон углов — спорный.** Раздатка организаторов говорит −90…+30°, полевой опыт
> другой команды — **−80…+30°** (`set_angle(-80)` ≈ надир). Разница в 10° на высоте 2 м
> даёт смещение ~35 см — больше, чем весь наш порог точности «<30 см». Проверить на борту
> первым делом и заложить фактический угол в `R_mount`
> ([DRONE_PLAN.md](DRONE_PLAN.md) §4.1, [lessons_from_archipelago.md](lessons_from_archipelago.md) §1.1).

### Минимальный рабочий пример (из документации Code OSS)

```python
from pioneer_sdk2 import Camera, ImageViewer

cam = Camera()
viewer = ImageViewer()
while True:
    img = cam.get_cv_frame()
    viewer.imshow("test", img)
```

---

## 6. Нейросети: Pioneer-RKNN

Предустановлена в Pioneer OS. Работает с NPU, формат моделей — **только `.rknn`**.

### Поддерживаемые семейства

| Класс | Архитектуры | Задача |
|---|---|---|
| `Yolo` | `yolov8`, `yolov11` | детекция и классификация |
| `YoloPose` | `yolov8-pose` | оценка позы |
| `PaddleOCR` | `PP-OCRv5_mobile` (detect + recognize) | распознавание текста |

### Классы

- **`ModelContainer`** — базовый класс инференса. У наследника есть атрибут `arch`
  (список поддерживаемых архитектур) и переопределённый `run()`.
- **`ModelRegistry`** — работа с реестром моделей по HTTP:
  `list_model()`, `get_model_info(name)`, `upload_model()`, `delete_model()`.
- **`Yolo`** — инференс + постобработка. Пороги: детекции `0.25`, NMS `0.45`.
- **`YoloPose`** — возвращает боксы + массив ключевых точек `[17×3]` (скелет).
- **`PaddleOCR`** — двухстадийный пайплайн: детекция текстовых областей →
  распознавание символов.

### Базовое использование

```python
from pioneer_rknn import Yolo

model = Yolo(model_name="yolov8")
```

### Загрузка своей модели

Через сервис **AI Models** (`http://172.17.49.2:7777/`) или меню
«Обучи и загрузи ИИ → AI Models». Можно: смотреть список, загружать, редактировать
метаданные, удалять.

Поле **«Архитектура»** при загрузке:
- если архитектура поддерживается — вписать её имя как при обучении (`yolov8`, `yolov8-pose`);
- если нет — вписать `custom`.

После регистрации модель доступна по имени из `pioneer_rknn`.

> **Пайплайн для нашего проекта**: обучаем YOLOv8/v11 на своих данных → конвертируем в
> `.rknn` → заливаем через `:7777` → грузим в коде через `Yolo(model_name="...")`.
>
> ⚠️ **На практике этот короткий путь не работает.** Полевой опыт другой команды: стандартный
> ONNX-экспорт (плоская голова `1×6×8400` со встроенным decode) после INT8-квантования даёт
> на борту **ноль детекций** — DFL-голова не квантуется. Рабочий путь: raw-branch экспорт
> (форк `airockchip`) → регистрация с архитектурой **`custom`** → decode руками на CPU.
> Подробно, с готовым кодом — [lessons_from_archipelago.md](lessons_from_archipelago.md) §2.

---

## 7. Code OSS — где писать код

`http://172.17.49.2:9999/` (или карточка «Пиши код как профи» в Pioneer Code).

- Создать файл: кнопка «Создать файл», не забыть расширение `.py`. Или двойной клик в explorer.
- Создать папку: «Создать папку» + Enter.
- Удалить: ПКМ → удалить, либо выделить и Delete.
- Запуск: иконка «play» в правом верхнем углу. Вывод — в терминал снизу.
- Остановка: красная иконка «stop».
- Есть встроенный терминал (`ls`, `cd`, `nano`, `sudo`, `mkdir`, `rm -rf`, `touch`).
  `nano`: Ctrl+S — сохранить, Ctrl+X — выйти, Ctrl+K — вырезать, Ctrl+U — вставить.

---

## 8. Эксплуатация: индикация и режимы

### Питание и заряд
Одна кнопка: короткое нажатие — показать заряд, длинное — вкл/выкл.

Цвет индикатора заряда: зелёный 67–100%, жёлтый 34–66%, красный 0–33%.

### Светодиоды состояния

| Индикация | Значение |
|---|---|
| Мигает оранжевый | загрузка |
| Мигает зелёный | готов к работе |
| Мигает фиолетовый | идёт обновление Linux |
| Красный/фиолетовый | ошибка автопилота |
| Красный/оранжевый | ошибка Linux |

### Режимы полёта

1. **STABILIZE** — только автовыравнивание по крену/тангажу
2. **ALTHOLD** — удержание высоты
3. **HEADLESS** — блокировка курса
4. **LOITER** — удержание позиции (по оптическому потоку)
5. **MISSION** — автономный полёт по программе

### Батарея и винты
- Батарея вставляется контактами вниз. Зарядка штатным USB — 1–1.5 ч.
- Защита: автоотключение при переразряде (<6 В) и перегрузке по току (>20 А).
- Винты: 2 CCW + 2 CW. У моторов правого вращения — выступающие метки, у
  соответствующих винтов маркировка **«R»**.

### Способы управления
1. Сервис Pioneer Code (программирование миссий) — **наш вариант**
2. Мобильное приложение Geoscan Jump 2
3. Пульт BetaFPV LiteRadio 2SE

---

## 9. Ссылки

- Главная Mini 2: https://docs.geoscan.ru/pioneer/instructions/pioneer-mini2/main-mini2.html
- Pioneer OS: `.../pioneer-mini2/pio_os/pio_os.html`
- Pioneer Code: `.../pioneer-mini2/pio_os/pioneer_code/pioneer_code.html`
- Code OSS: `.../pioneer-mini2/pio_os/pioneer_code/code-oss.html`
- AI Models: `.../pioneer-mini2/pio_os/pioneer_code/ai-models.html`
- Основы Linux: `.../pioneer-mini2/pio_os/linux-basics.html`
- Настройки: `.../pioneer-mini2/settings.html`
- Pioneer SDK2: https://docs.geoscan.ru/pioneer/programming/python/pio-sdk2.html
- Pioneer-RKNN: https://docs.geoscan.ru/pioneer/programming/python/pio-rknn.html
- Поддержка: https://t.me/geoscan_edu, support@geoscan.ru

---

## Оговорка

Сигнатуры методов SDK2 и RKNN собраны пересказом с сайта, а не из исходников.
Перед тем как закладываться на конкретный порядок аргументов или значение по
умолчанию — проверить на борту:

```python
import pioneer_sdk2, pioneer_rknn
help(pioneer_sdk2.Pioneer)
help(pioneer_rknn.Yolo)
```

Раздел «Настройки» на сайте помечен как «в процессе доработки» — параметры
автопилота, geofence, калибровка и прошивка там пока не описаны. Их надо будет
добрать отдельно (вероятно, из документации Pioneer Station 2.0).
