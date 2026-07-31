#!/usr/bin/env bash
# Заливка бортового ПО на квадрокоптер.
#
#   ./push_to_drone.sh                 залить и снести __pycache__
#   ./push_to_drone.sh --run           залить и сразу запустить main.py
#   DRONE=172.17.49.2 ./push_to_drone.sh
#
# Процедура повторяет то, что сработало у команды с этим же железом
# (docs/lessons_from_archipelago.md §5). Два пункта там неочевидные и оба важные:
#
#   - __pycache__ на борту надо СНОСИТЬ после каждой заливки, иначе Python возьмёт
#     старый .pyc, и вы будете час искать, почему исправленный код ведёт себя по-старому;
#   - никаких venv: в виртуальном окружении pioneer_sdk2 не работает.
set -eu

# 172.17.49.2 — адрес нашего борта, проверен на площадке 31.07.2026. В документации
# Geoscan и у другой команды фигурирует 10.42.0.1 — у нашего экземпляра это не так.
DRONE="${DRONE:-172.17.49.2}"
USER_NAME="${DRONE_USER:-pioneermini}"
DEST="${DRONE_DIR:-/home/pioneermini/workspace}"
КОРЕНЬ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== Заливка на ${USER_NAME}@${DRONE}:${DEST} =="

ssh "${USER_NAME}@${DRONE}" "mkdir -p ${DEST}/protocol ${DEST}/tools"

# Бортовые модули кладутся плоско: на борту main.py лежит рядом со своими соседями,
# а не в директории с русским именем — так проще звать из Code OSS и из ssh.
scp "${КОРЕНЬ}/квадрокоптер/"*.py    "${USER_NAME}@${DRONE}:${DEST}/"
scp "${КОРЕНЬ}/квадрокоптер/config.yaml" "${USER_NAME}@${DRONE}:${DEST}/"
scp "${КОРЕНЬ}/распределительный хаб/protocol/"*.py "${USER_NAME}@${DRONE}:${DEST}/protocol/"
# Хаб и передатчик — тоже на борт. Штатно они живут на компьютере оператора, но если
# на выданной машине нет прав администратора, брандмауэр Windows не пропустит входящее
# соединение на порт 5001, и команда на взлёт до дрона не дойдёт. Хаб на борту снимает
# вопрос целиком: исходящие соединения Windows не блокирует.
scp "${КОРЕНЬ}/распределительный хаб/hub.py" "${КОРЕНЬ}/распределительный хаб/transmitter.py" \
    "${USER_NAME}@${DRONE}:${DEST}/"
scp "${КОРЕНЬ}/tools/record.py" "${КОРЕНЬ}/tools/check_sdk.py" "${КОРЕНЬ}/tools/check_axes.py" \
    "${USER_NAME}@${DRONE}:${DEST}/tools/"

# main.py сам находит protocol/ и tools/ в обеих раскладках — плоской и репозиторной.
ssh "${USER_NAME}@${DRONE}" "cd ${DEST} && rm -rf __pycache__ protocol/__pycache__ tools/__pycache__ && ls -la"

echo
echo "Залито. Запуск на борту:"
echo "  ssh ${USER_NAME}@${DRONE}"
echo "  cd ${DEST} && python3 main.py --hub http://<адрес хаба>:5001"
echo
echo "Если хаб держим на борту (нет прав администратора на ноутбуке):"
echo "  окно 1: ssh ${USER_NAME}@${DRONE} 'cd ${DEST} && python3 hub.py'"
echo "  окно 2: ssh ${USER_NAME}@${DRONE} 'cd ${DEST} && python3 main.py --hub http://127.0.0.1:5001'"
echo "  с ноутбука: python 'распределительный хаб/transmitter.py' --hub http://${DRONE}:5001 takeoff"

if [ "${1:-}" = "--run" ]; then
    ssh -t "${USER_NAME}@${DRONE}" "cd ${DEST} && python3 main.py"
fi
