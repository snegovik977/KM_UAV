#!/usr/bin/env bash
# ============================================================================
#  СТАБИЛЬНАЯ ВЕРСИЯ v1.1 -> дрон
#  Заливка known-good снапшота на борт. Запускать с Mac из папки stable_v1.1/.
#
#    cd stable_v1.1
#    ./push_to_drone.sh            # хост по умолчанию: pioneermini@10.42.0.1
#    ./push_to_drone.sh 192.168.4.1   # или явно указать IP борта
#
#  Раскладка этой папки уже совпадает с ~/workspace на дроне:
#    autonomous_mission_ai.py, vision_node.py, vision/, onboard/
# ============================================================================
set -e

DRONE="${1:-pioneermini@10.42.0.1}"
DEST="/home/pioneermini/workspace"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo ">>> Заливка стабильной версии v1.1 на ${DRONE}:${DEST}"

# полётный файл + vision-нода
scp "${HERE}/autonomous_mission_ai.py" "${HERE}/vision_node.py" \
    "${DRONE}:${DEST}/"

# пакеты vision/ и onboard/ (без кэша), с удалением лишнего на борту
rsync -av --delete --exclude='__pycache__' \
    "${HERE}/vision" "${HERE}/onboard" \
    "${DRONE}:${DEST}/"

# модель (ПЛОСКИЙ FP16 — под плоский detector.py этой версии) в ~/workspace
RKNN="$(ls "${HERE}"/*.rknn | head -1)"
scp "${RKNN}" "${DRONE}:${DEST}/"

# на борту сносим старый .pyc, чтобы не подхватился устаревший байткод
ssh "${DRONE}" "cd ${DEST} && rm -rf __pycache__ vision/__pycache__ onboard/__pycache__"

echo ">>> Код и модель залиты."
echo ">>> ОДИН РАЗ зарегистрировать модель в «Модели ИИ»:"
echo "     - удалить прежнюю 'yolo11nnew';"
echo "     - загрузить $(basename "${RKNN}"), имя 'yolo11nnew', архитектура 'custom'."
echo ">>> Запуск на борту:"
echo "    ssh ${DRONE}"
echo "    cd ${DEST} && python3 autonomous_mission_ai.py"
