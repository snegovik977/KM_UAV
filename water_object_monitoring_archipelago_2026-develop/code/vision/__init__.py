"""Пакет боевой логики зрения: детекция -> цвет -> ArUco -> дедуп/подсчёт -> оверлей/лог."""
from .detector import Detection, ColorBlobDetector, YoloRknnDetector
from .color import classify_color
from .pool_roi import pool_mask, center_in_mask, pool_coverage
from .aruco_id import ArucoReader
from .tracker import CentroidTracker
from .pipeline import ShipMonitor
