#!/usr/bin/env bash
# ============================================================================
#  СТАБИЛЬНАЯ ВЕРСИЯ v1.2 -> дрон   (лучшая на текущий момент)
#  Отличия v1.2 от v1.1:
#    - нейросеть INT8 raw-branch (ships_v2_rk3576_int8raw.rknn) + новый detector.py;
#    - ROI бассейна ВЫКЛ, детект каждый кадр (быстрое обнаружение, как v1.0);
#    - классификация в терминал РАЗВЯЗАНА с ArUco (не теряем +20 при нечитаемой метке);
#    - запись видеопотока в память борта.
#
#  Запускать с Mac из папки stable_v1.2/:
#    ./push_to_drone.sh                 # хост по умолчанию: pioneermini@10.42.0.1
#    ./push_to_drone.sh 192.168.4.1     # или явный IP борта
#
#  ⚠️ Модель raw-branch и detector.py — ПАРА. Не смешивать с плоской моделью (v1.0/v1.1).
# ============================================================================
set -e

DRONE="${1:-pioneermini@10.42.0.1}"
DEST="/home/pioneermini/workspace"
HERE="$(cd "$(dirname "$0")" && pwd)"
RKNN="$(ls "${HERE}"/*.rknn | head -1)"

echo ">>> Заливка стабильной версии v1.2 на ${DRONE}:${DEST}"

# полётный файл + vision-нода
scp "${HERE}/autonomous_mission_ai.py" "${HERE}/vision_node.py" \
    "${DRONE}:${DEST}/"

# пакеты vision/ и onboard/ (без кэша), с удалением лишнего на борту
rsync -av --delete --exclude='__pycache__' \
    "${HERE}/vision" "${HERE}/onboard" \
    "${DRONE}:${DEST}/"

# модель кладём в ~/workspace (регистрацию см. ниже)
scp "${RKNN}" "${DRONE}:${DEST}/"

# на борту сносим старый .pyc
ssh "${DRONE}" "cd ${DEST} && rm -rf __pycache__ vision/__pycache__ onboard/__pycache__"

echo ">>> Код и модель залиты."
echo ">>> ОДИН РАЗ зарегистрировать модель в реестре «Модели ИИ»:"
echo "     - удалить прежнюю 'yolo11nnew';"
echo "     - загрузить $(basename "${RKNN}"), имя 'yolo11nnew', архитектура 'custom'."
echo ">>> Запуск: ssh ${DRONE} ; cd ${DEST} ; python3 autonomous_mission_ai.py"
