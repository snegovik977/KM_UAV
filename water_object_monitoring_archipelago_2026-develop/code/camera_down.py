from pioneer_sdk2 import Pioneer, ServoCamera

# Подключение к дрону
pioneer = Pioneer()

# Опускаем камеру максимально вниз (диапазон -80..+30, -80 = почти надир)
servo = ServoCamera()

try:
    ok = servo.set_angle(-80)
    print(f"set_angle(-80) -> {ok}")
except Exception as e:
    print(f"Не удалось наклонить камеру: {e}")

finally:
    pioneer.close_connection()
    print("Готово.")
