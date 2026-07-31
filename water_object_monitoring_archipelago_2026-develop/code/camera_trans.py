from pioneer_sdk2 import Pioneer, Camera, ImageViewer, CameraType

# Подключение к дрону
pioneer = Pioneer()

# Инициализация камеры и viewer
camera = Camera(camera_type=CameraType.MAIN)
viewer = ImageViewer()

try:
    while True:
        # Получение кадра с камеры
        frame = camera.get_cv_frame(timeout=5.0)

        if frame is not None:
            # Отправка кадра в веб-трансляцию
            viewer.imshow("main_camera", frame, fps=30)

finally:
    # Остановка камеры и закрытие соединения
    camera.stop()
    viewer.close()
    pioneer.close_connection()
