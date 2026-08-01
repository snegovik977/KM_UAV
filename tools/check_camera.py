#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Почему камера не отдаёт кадров. Запускается НА БОРТУ, полёта не требует.

Симптом, ради которого написан:

    shm_open failed on /shmpipe. ... No such file or directory
    TimeoutError: Не удалось получить кадр в течение 5.0 секунд

Драйвер gstreamer забирает кадры не с /dev/video напрямую, а из shared memory,
куда их кладёт отдельная служба (media-server). `/shmpipe` не существует — значит
писать в него некому. Причин ровно три, и вслепую они неразличимы:

  1. служба камеры не запущена или упала;
  2. камеру уже держит другой процесс — забытый main.py или handheld.py
     в соседнем ssh-окне (камера отдаётся ОДНОМУ потребителю);
  3. драйвер gstreamer на этом борту не поднимается вовсе, и нужен второй,
     RTSP (документация Geoscan называет оба, см. docs/pioneer_mini2.md §5).

Скрипт отвечает, какая из трёх, и пробует получить кадр каждым доступным драйвером.
Ничего не чинит и не перезапускает сам: на площадке решение о рестарте службы
принимает человек.

    python3 tools/check_camera.py
    python3 tools/check_camera.py > camera_report.txt 2>&1
"""
from __future__ import annotations

import os
import subprocess
import sys
import traceback


def заголовок(текст):
    print("")
    print("=" * 72)
    print(текст)
    print("=" * 72)


def выполнить(команда, пояснение=""):
    """Выполнить команду оболочки и напечатать вывод. Не падает никогда."""
    print("")
    print("$ %s%s" % (команда, ("      # " + пояснение) if пояснение else ""))
    try:
        вывод = subprocess.run(команда, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=20)
        текст = вывод.stdout.decode("utf-8", "replace").strip()
        print(текст if текст else "(пусто)")
    except Exception as e:
        print("не выполнилось: %s: %s" % (type(e).__name__, e))


def раздел_процессы():
    заголовок("1. Кто ещё держит камеру")
    print("Камера отдаётся ЧЕРЕЗ SHARED MEMORY и только ОДНОМУ потребителю.")
    print("Забытый в другом ssh-окне main.py выглядит ровно как «камера сломалась».")
    выполнить("pgrep -af 'python3?' || true", "все питоновские процессы")
    выполнить("fuser -v /dev/shm/* 2>&1 | head -40 || true", "кто открыл shm")


def раздел_устройства():
    заголовок("2. Устройства и shared memory")
    print("/shmpipe должен лежать в /dev/shm, пока служба камеры работает.")
    выполнить("ls -l /dev/video* 2>&1 || true", "камеры ядра")
    выполнить("ls -la /dev/shm/ 2>&1 || true", "сюда пишет media-server")


def раздел_службы():
    заголовок("3. Служба камеры / media-server")
    print("Если служба не active — кадров не будет ни у кого, и это главная причина.")
    выполнить("systemctl list-units --type=service --all --no-pager "
              "| grep -i -E 'cam|media|mtx|rtsp|gst|pioneer' || true")
    выполнить("systemctl --no-pager --lines=15 status "
              "$(systemctl list-units --type=service --all --plain --no-legend "
              "| grep -i -E 'cam|media|mtx|rtsp' | awk '{print $1}' | head -3) 2>&1 "
              "| head -60 || true", "подробности по найденным")
    print("")
    print("Если служба лежит, поднять её обычно можно так (имя подставить своё):")
    print("    sudo systemctl restart <имя-службы>")
    print("Перезагрузка борта тоже лечит, но стоит минуты — сначала посмотрите статус.")


def раздел_порты():
    заголовок("4. Порты трансляции")
    print("8554 RTSP, 8888 HLS, 8889 WebRTC — раскладка mediamtx. Что из этого")
    print("реально слушает, столько и живо на борту.")
    выполнить("(ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | head -30 || true")


def раздел_mediamtx():
    заголовок("4а. Что думает сам mediamtx")
    print("У mediamtx есть HTTP API (порт 9997). Он скажет, какие пути настроены")
    print("и есть ли у них ИСТОЧНИК. «ready: false» = публиковать некому.")
    выполнить("curl -s --max-time 5 http://127.0.0.1:9997/v3/paths/list || true")
    выполнить("curl -s --max-time 5 http://127.0.0.1:9997/v3/config/paths/list || true",
              "как пути настроены")


def раздел_сенсор():
    заголовок("4б. Отдаёт ли кадры сам сенсор")
    print("Конвейер media-server берёт кадры с /dev/video44 (main) и /dev/video47 (opt) —")
    print("номера видно в `systemctl status media-server`, они могут отличаться.")
    print("Если сырой захват отсюда не идёт, дело не в SDK и не в shared memory.")
    выполнить("v4l2-ctl --list-devices 2>&1 | head -40 || true")
    выполнить("timeout 15 gst-launch-1.0 -v v4l2src device=/dev/video44 num-buffers=5 "
              "! video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 "
              "! fakesink 2>&1 | tail -20 || true",
              "5 кадров напрямую с сенсора")
    выполнить("dmesg 2>/dev/null | grep -iE 'rkisp|rkcif|video|camera|imx|ov[0-9]{4}' "
              "| tail -25 || true", "что ядро говорит про камеру")


def раздел_sdk():
    заголовок("5. Что SDK умеет по камере")
    try:
        import inspect

        import pioneer_sdk2
        from pioneer_sdk2 import Camera, CameraType
    except Exception as e:
        print("pioneer_sdk2 не импортируется: %s: %s" % (type(e).__name__, e))
        return None, None, []

    print("pioneer_sdk2: %s" % getattr(pioneer_sdk2, "__file__", "?"))
    try:
        print("Camera.__init__%s" % inspect.signature(Camera.__init__))
    except Exception as e:
        print("сигнатуру Camera прочитать не вышло: %s" % e)
    try:
        print("CameraType: %s" % [и for и in dir(CameraType) if not и.startswith("_")])
    except Exception:
        pass

    # Какие драйверы вообще есть. Имя пакета взято из сообщения на борту:
    # «ImageViewer из SDK поднялся: pioneer_sdk2.camera_driver.gstreamer».
    драйверы = []
    try:
        import pkgutil

        import pioneer_sdk2.camera_driver as пакет
        путь = os.path.dirname(пакет.__file__)
        print("")
        print("пакет драйверов: %s" % путь)
        for _, имя, _ in pkgutil.iter_modules([путь]):
            драйверы.append(имя)
        print("драйверы: %s" % (драйверы or "не нашлось"))
    except Exception as e:
        print("камерные драйверы не перечислились: %s: %s" % (type(e).__name__, e))

    # Чем именно выбирается драйвер — ключевой вопрос, если gstreamer не оживить.
    # ✅ Проверено 01.08.2026: Camera.__init__(self, camera_type) и БОЛЬШЕ НИЧЕГО,
    # а драйверы в пакете лежат оба (gstreamer, rtsp). Значит выбор спрятан внутри —
    # переменная окружения, файл настроек или жёстко зашит. Ищем в исходниках SDK.
    print("")
    print("как выбирается драйвер (в сигнатуре Camera его нет — ищем внутри SDK):")
    корень_sdk = os.path.dirname(pioneer_sdk2.__file__)
    выполнить("grep -rn --include=*.py -iE "
              "\"rtsp|gstreamer|getenv|environ|driver *=\" %s | head -30 || true"
              % корень_sdk)
    print("")
    print("Если найдётся переменная окружения — драйвер переключается так:")
    print("    ИМЯ_ПЕРЕМЕННОЙ=rtsp python3 tools/handheld.py")
    print("RTSP-путь у mediamtx: rtsp://127.0.0.1:8554/<имя>, порт 8554 слушает.")
    return Camera, CameraType, драйверы


def раздел_кадр(Camera, CameraType, драйверы):
    заголовок("6. Попытка получить кадр")
    if Camera is None:
        print("SDK не импортировался — пробовать нечего")
        return

    import inspect
    try:
        параметры = list(inspect.signature(Camera.__init__).parameters)
    except Exception:
        параметры = []
    ключ = None
    for кандидат in ("driver", "camera_driver", "backend", "driver_type"):
        if кандидат in параметры:
            ключ = кандидат
            break

    попытки = [("по умолчанию", {})]
    if ключ:
        print("драйвер выбирается параметром %r" % ключ)
        for имя in драйверы:
            if имя.startswith("_") or служебное_имя(имя):
                continue
            попытки.append((имя, {ключ: имя}))
    else:
        print("параметра выбора драйвера в сигнатуре нет — пробуем только умолчание")

    for метка, добавка in попытки:
        print("")
        print("--- драйвер: %s" % метка)
        камера = None
        try:
            аргументы = {"camera_type": CameraType.MAIN}
            аргументы.update(добавка)
            камера = Camera(**аргументы)
            кадр = камера.get_cv_frame(timeout=5.0)
            if кадр is None:
                print("    камера открылась, но кадр пустой (None)")
            else:
                print("    ✅ КАДР ЕСТЬ: %s, тип %s"
                      % (getattr(кадр, "shape", "?"), getattr(кадр, "dtype", "?")))
        except Exception as e:
            print("    не вышло: %s: %s" % (type(e).__name__, e))
        finally:
            try:
                if камера is not None:
                    камера.stop()
            except Exception:
                pass


def служебное_имя(имя):
    return имя in ("base", "utils", "common", "__init__")


def main():
    print("Диагностика камеры борта. Хост: %s, Python %s"
          % (os.uname().nodename if hasattr(os, "uname") else "?",
             sys.version.split()[0]))
    for раздел in (раздел_процессы, раздел_устройства, раздел_службы, раздел_порты,
                   раздел_mediamtx, раздел_сенсор):
        try:
            раздел()
        except Exception:
            traceback.print_exc()
    try:
        Camera, CameraType, драйверы = раздел_sdk()
        раздел_кадр(Camera, CameraType, драйверы)
    except Exception:
        traceback.print_exc()

    заголовок("Что делать по результатам")
    print("Кадр есть хоть одним драйвером      -> запускать handheld.py")
    print("Служба не active                    -> поднять её (раздел 3)")
    print("Чужой python висит                  -> убить: pkill -f main.py")
    print("/dev/video пуст                     -> шлейф камеры или железо")
    print("")
    print("СЛУЖБА ЖИВА, а /dev/shm ПУСТ и mediamtx пишет 'no stream available' —")
    print("значит конвейер запущен, но сенсор не отдал НИ ОДНОГО кадра. Порядок:")
    print("  1. sudo systemctl restart media-server   # дёшево, лечит чаще всего")
    print("     затем заново python3 tools/check_camera.py")
    print("  2. если не помогло — смотреть раздел 4б: пошёл ли сырой захват")
    print("     с /dev/video44. Не пошёл -> проблема ниже SDK, в сенсоре;")
    print("     пошёл, а shm пуст -> перезапускать именно скрипты shm:")
    print("     sudo pkill -f start_maincamera_shm.sh   (служба поднимет заново)")
    print("  3. крайнее средство — sudo reboot, минута времени")
    return 0


if __name__ == "__main__":
    sys.exit(main())
