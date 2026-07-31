#!/usr/bin/env bash
# ============================================================================
#  НОВАЯ ВЕРСИЯ (INT8 raw-branch + правки ArUco + запись видео) -> дрон
#  Запускать с Mac из корня репозитория:
#
#    ./push_new_version.sh                 # хост по умолчанию: pioneermini@10.42.0.1
#    ./push_new_version.sh 192.168.4.1     # или явный IP борта
#
#  Заливает СРАЗУ ВСЁ согласованным набором (иначе краш из-за рассинхрона):
#    - drone/autonomous_mission_ai.py, drone/vision_node.py  (vision_node с record=)
#    - code/vision/*   (raw-branch detector.py + правки aruco_id.py/pipeline.py)
#    - code/onboard/runtime.py
#    - модель INT8 raw-branch .rknn (в ~/workspace; регистрацию см. в конце)
#
#  ⚠️ Модель и detector.py — ПАРА (raw-branch). Плоская модель + новый detector = мусор.
#     Откат при проблемах: cd stable_v1.1 && ./push_to_drone.sh
# ============================================================================
set -e

DRONE="${1:-pioneermini@10.42.0.1}"
DEST="/home/pioneermini/workspace"
HERE="$(cd "$(dirname "$0")" && pwd)"
RKNN="rknn_convert/ships_v2_rk3576_int8raw.rknn"

echo ">>> Заливка НОВОЙ версии на ${DRONE}:${DEST}"

# 1) полётный файл + vision-нода (vision_node.py с параметром record -> чинит краш)
scp "${HERE}/drone/autonomous_mission_ai.py" "${HERE}/drone/vision_node.py" \
    "${DRONE}:${DEST}/"

# 2) пакеты vision/ и onboard/ (raw-branch detector + правки ArUco), без кэша
rsync -av --delete --exclude='__pycache__' \
    "${HERE}/code/vision" "${HERE}/code/onboard" \
    "${DRONE}:${DEST}/"

# 3) модель INT8 raw-branch — кладём файл в ~/workspace (регистрация — отдельным шагом)
scp "${HERE}/${RKNN}" "${DRONE}:${DEST}/"

# 4) на борту сносим старый .pyc
ssh "${DRONE}" "cd ${DEST} && rm -rf __pycache__ vision/__pycache__ onboard/__pycache__"

echo ">>> Код и модель залиты."
echo ">>> ОСТАЛОСЬ ОДИН РАЗ зарегистрировать модель в реестре «Модели ИИ»:"
echo "     - удалить старую 'yolo11nnew' (плоский FP16);"
echo "     - загрузить ships_v2_rk3576_int8raw.rknn, имя 'yolo11nnew', архитектура 'custom'."
echo "   (веб-UI Pioneer Code, либо ModelRegistry().upload_model на борту)."
echo ">>> Затем наземный тест:  ssh ${DRONE} ; cd ${DEST} ; python3 vision_node.py --seconds 30"
