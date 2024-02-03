import numpy as np
import torch
from scipy.optimize import minimize
import os
import ctypes
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QBrush, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from utils import get_face_landmarks_in_ccs, gaze_2d_to_3d, get_camera_matrix, ray_plane_intersection, get_point_on_screen, plane_equation
from mpii_face_gaze_preprocessing import normalize_single_image
from model import Model, RegressionMLP, RegressionRNN
from torch import nn
import mediapipe as mp
from torch.utils.data import Dataset, DataLoader
import cv2
import time
import warnings
warnings.filterwarnings("ignore")

# Global variables for circle parameters
circle_radius = 50
outline_width = 2
outline_color = Qt.white


def hybrid_method(cam_eye_point, target_screen_point, gaze_pitch, gaze_yaw):
    """
    Hybrid calibration for eye gaze estimation.
    :param cor_eye: matrix of eye corneal reflection coordinates
    :param target_cor: atrix of target coordinates
    :return: optimal value and optimal solution
    """

    # 创建一个角度变量p
    p = 0

    # 创建三个变量x, y, z
    x, y, z = 0, 0, 0

    # 定义目标函数
    def objective_function(variables):
        p, x, y, z = variables
        rotation_mat = np.array([[1, 0, 0], [0, np.cos(np.radians(p)), -np.sin(np.radians(p))],
                                 [0, np.sin(np.radians(p)), np.cos(np.radians(p))]])
        translation_mat = np.array([x, y, z])
        screen_eye_point = np.dot(rotation_mat, cam_eye_point).T + translation_mat

        x_screen = screen_eye_point[:, -1] * np.tan(gaze_yaw) + screen_eye_point[:, 0]
        y_screen = screen_eye_point[:, -1] * np.tan(gaze_pitch + p) + screen_eye_point[:, 1]


        predict_cor = np.vstack((x_screen, y_screen)).T

        return np.sum((predict_cor - target_screen_point) ** 2)

    # 初始猜测值
    initial_guess = [p, x, y, z]

    # 最小化目标函数
    result = minimize(objective_function, initial_guess)

    return result.x, result.fun / cam_eye_point.shape[1]


def get_monitor_dimensions():
    user32 = ctypes.windll.user32

    # 获取屏幕尺寸
    width_pixels = user32.GetSystemMetrics(0)  # 0 表示屏幕宽度
    height_pixels = user32.GetSystemMetrics(1)  # 1 表示屏幕高度

    hdc = user32.GetDC(0)

    # 尝试使用 GetDpiForWindow 获取 DPI
    hwnd = user32.GetDesktopWindow()
    dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)

    # 如果 GetDpiForWindow 不可用，尝试使用 GetDeviceCaps 获取 DPI
    if dpi == 0:
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # 88 表示 LOGPIXELSX，表示水平像素数

    width_mm = width_pixels / (dpi / 25.4)  # 计算宽度（毫米）
    height_mm = height_pixels / (dpi / 25.4)  # 计算高度（毫米）

    user32.ReleaseDC(0, hdc)

    return (width_mm, height_mm), (width_pixels, height_pixels)

class GameOverlay(QWidget):
    def __init__(self, center_coordinates, delay):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()

        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        self.setGeometry(0, 0, screen_width, screen_height)  # Set the overlay size to match the screen
        self.center_coordinates = center_coordinates  # List of center coordinates
        self.delay = delay
        self.show()
        self.current_circle_index = 0  # Initialize index for the current circle

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw a transparent background
        painter.setBrush(QBrush(QColor(0, 0, 0, 0)))
        painter.drawRect(self.rect())

        # Check if there are more circles to draw
        if self.current_circle_index < len(self.center_coordinates):
            center_x, center_y = self.center_coordinates[self.current_circle_index]

            # Draw the circle outline for the current coordinates
            painter.setPen(QColor(outline_color))
            painter.setBrush(QBrush(QColor(0, 0, 0, 0)))
            painter.drawEllipse(center_x - circle_radius, center_y - circle_radius,
                                circle_radius * 2, circle_radius * 2)

            self.current_circle_index += 1
            if self.current_circle_index >= len(self.center_coordinates):
                # If all circles are drawn, quit the application
                QApplication.quit()
            else:
                # If there are more circles, schedule a repaint after a delay
                QTimer.singleShot(self.delay, self.update)  # Update after a 1000 ms delay

class myDataset(Dataset):
    def __init__(self, eye_points, fix_pos):
        self.data = eye_points
        self.fix_pos = fix_pos

    def __getitem__(self, index):
        return self.data[index], self.fix_pos[index]

    def __len__(self):
        return len(self.data)

class screen_camera_calibration():

    def __init__(self, camera_matrix, dist_coefficients, finetune=True):
        frames_path = os.listdir('./data')

        self.regression = RegressionMLP()
        if finetune:
            try:
                weight = torch.load('regression_model.pth')
                self.regression.load_state_dict(weight)
            except:
                pass
            p = -1
            self.frames = [cv2.imread('./data/{}/{}'.format(frames_path[p], frame)) for frame in
                           sorted([each for each in os.listdir('./data/{}'.format(frames_path[p])) if each.endswith('jpg')], key=lambda x: int(x.split('.jpg')[0]))]
            self.fix_pos = [open('./data/{}/{}'.format(frames_path[p], 'screen_position.txt'), encoding='utf8').readlines()]
        else:
            self.frames = [cv2.imread('./data/{}/{}'.format(each,frame_path)) for each in frames_path for frame_path in sorted(os.listdir('./data/{}'.format(frames_path[-1]))[:-1], key=lambda x: int(x.split('.jpg')[0]))]
            self.fix_pos = [open('./data/{}/{}'.format(each, 'screen_position.txt')).readlines() for each in frames_path]

        for i in range(len(self.fix_pos)):
            self.fix_pos[i] = np.array([list(map(int, each.replace('\n','').split(' '))) for each in self.fix_pos[i]])
        self.fix_pos = np.concatenate(self.fix_pos)
        self.app = QApplication(sys.argv)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.face_model_all: np.ndarray = np.array([
            [0.000000, -3.406404, 5.979507],
            [0.000000, -1.126865, 7.475604],
            [0.000000, -2.089024, 6.058267],
            [-0.463928, 0.955357, 6.633583],
            [0.000000, -0.463170, 7.586580],
            [0.000000, 0.365669, 7.242870],
            [0.000000, 2.473255, 5.788627],
            [-4.253081, 2.577646, 3.279702],
            [0.000000, 4.019042, 5.284764],
            [0.000000, 4.885979, 5.385258],
            [0.000000, 8.261778, 4.481535],
            [0.000000, -3.706811, 5.864924],
            [0.000000, -3.918301, 5.569430],
            [0.000000, -3.994436, 5.219482],
            [0.000000, -4.542400, 5.404754],
            [0.000000, -4.745577, 5.529457],
            [0.000000, -5.019567, 5.601448],
            [0.000000, -5.365123, 5.535441],
            [0.000000, -6.149624, 5.071372],
            [0.000000, -1.501095, 7.112196],
            [-0.416106, -1.466449, 6.447657],
            [-7.087960, 5.434801, 0.099620],
            [-2.628639, 2.035898, 3.848121],
            [-3.198363, 1.985815, 3.796952],
            [-3.775151, 2.039402, 3.646194],
            [-4.465819, 2.422950, 3.155168],
            [-2.164289, 2.189867, 3.851822],
            [-3.208229, 3.223926, 4.115822],
            [-2.673803, 3.205337, 4.092203],
            [-3.745193, 3.165286, 3.972409],
            [-4.161018, 3.059069, 3.719554],
            [-5.062006, 1.934418, 2.776093],
            [-2.266659, -7.425768, 4.389812],
            [-4.445859, 2.663991, 3.173422],
            [-7.214530, 2.263009, 0.073150],
            [-5.799793, 2.349546, 2.204059],
            [-2.844939, -0.720868, 4.433130],
            [-0.711452, -3.329355, 5.877044],
            [-0.606033, -3.924562, 5.444923],
            [-1.431615, -3.500953, 5.496189],
            [-1.914910, -3.803146, 5.028930],
            [-1.131043, -3.973937, 5.189648],
            [-1.563548, -4.082763, 4.842263],
            [-2.650112, -5.003649, 4.188483],
            [-0.427049, -1.094134, 7.360529],
            [-0.496396, -0.475659, 7.440358],
            [-5.253307, 3.881582, 3.363159],
            [-1.718698, 0.974609, 4.558359],
            [-1.608635, -0.942516, 5.814193],
            [-1.651267, -0.610868, 5.581319],
            [-4.765501, -0.701554, 3.534632],
            [-0.478306, 0.295766, 7.101013],
            [-3.734964, 4.508230, 4.550454],
            [-4.588603, 4.302037, 4.048484],
            [-6.279331, 6.615427, 1.425850],
            [-1.220941, 4.142165, 5.106035],
            [-2.193489, 3.100317, 4.000575],
            [-3.102642, -4.352984, 4.095905],
            [-6.719682, -4.788645, -1.745401],
            [-1.193824, -1.306795, 5.737747],
            [-0.729766, -1.593712, 5.833208],
            [-2.456206, -4.342621, 4.283884],
            [-2.204823, -4.304508, 4.162499],
            [-4.985894, 4.802461, 3.751977],
            [-1.592294, -1.257709, 5.456949],
            [-2.644548, 4.524654, 4.921559],
            [-2.760292, 5.100971, 5.015990],
            [-3.523964, 8.005976, 3.729163],
            [-5.599763, 5.715470, 2.724259],
            [-3.063932, 6.566144, 4.529981],
            [-5.720968, 4.254584, 2.830852],
            [-6.374393, 4.785590, 1.591691],
            [-0.672728, -3.688016, 5.737804],
            [-1.262560, -3.787691, 5.417779],
            [-1.732553, -3.952767, 5.000579],
            [-1.043625, -1.464973, 5.662455],
            [-2.321234, -4.329069, 4.258156],
            [-2.056846, -4.477671, 4.520883],
            [-2.153084, -4.276322, 4.038093],
            [-0.946874, -1.035249, 6.512274],
            [-1.469132, -4.036351, 4.604908],
            [-1.024340, -3.989851, 4.926693],
            [-0.533422, -3.993222, 5.138202],
            [-0.769720, -6.095394, 4.985883],
            [-0.699606, -5.291850, 5.448304],
            [-0.669687, -4.949770, 5.509612],
            [-0.630947, -4.695101, 5.449371],
            [-0.583218, -4.517982, 5.339869],
            [-1.537170, -4.423206, 4.745470],
            [-1.615600, -4.475942, 4.813632],
            [-1.729053, -4.618680, 4.854463],
            [-1.838624, -4.828746, 4.823737],
            [-2.368250, -3.106237, 4.868096],
            [-7.542244, -1.049282, -2.431321],
            [0.000000, -1.724003, 6.601390],
            [-1.826614, -4.399531, 4.399021],
            [-1.929558, -4.411831, 4.497052],
            [-0.597442, -2.013686, 5.866456],
            [-1.405627, -1.714196, 5.241087],
            [-0.662449, -1.819321, 5.863759],
            [-2.342340, 0.572222, 4.294303],
            [-3.327324, 0.104863, 4.113860],
            [-1.726175, -0.919165, 5.273355],
            [-5.133204, 7.485602, 2.660442],
            [-4.538641, 6.319907, 3.683424],
            [-3.986562, 5.109487, 4.466315],
            [-2.169681, -5.440433, 4.455874],
            [-1.395634, 5.011963, 5.316032],
            [-1.619500, 6.599217, 4.921106],
            [-1.891399, 8.236377, 4.274997],
            [-4.195832, 2.235205, 3.375099],
            [-5.733342, 1.411738, 2.431726],
            [-1.859887, 2.355757, 3.843181],
            [-4.988612, 3.074654, 3.083858],
            [-1.303263, 1.416453, 4.831091],
            [-1.305757, -0.672779, 6.415959],
            [-6.465170, 0.937119, 1.689873],
            [-5.258659, 0.945811, 2.974312],
            [-4.432338, 0.722096, 3.522615],
            [-3.300681, 0.861641, 3.872784],
            [-2.430178, 1.131492, 4.039035],
            [-1.820731, 1.467954, 4.224124],
            [-0.563221, 2.307693, 5.566789],
            [-6.338145, -0.529279, 1.881175],
            [-5.587698, 3.208071, 2.687839],
            [-0.242624, -1.462857, 7.071491],
            [-1.611251, 0.339326, 4.895421],
            [-7.743095, 2.364999, -2.005167],
            [-1.391142, 1.851048, 4.448999],
            [-1.785794, -0.978284, 4.850470],
            [-4.670959, 2.664461, 3.084075],
            [-1.333970, -0.283761, 6.097047],
            [-7.270895, -2.890917, -2.252455],
            [-1.856432, 2.585245, 3.757904],
            [-0.923388, 0.073076, 6.671944],
            [-5.000589, -6.135128, 1.892523],
            [-5.085276, -7.178590, 0.714711],
            [-7.159291, -0.811820, -0.072044],
            [-5.843051, -5.248023, 0.924091],
            [-6.847258, 3.662916, 0.724695],
            [-2.412942, -8.258853, 4.119213],
            [-0.179909, -1.689864, 6.573301],
            [-2.103655, -0.163946, 4.566119],
            [-6.407571, 2.236021, 1.560843],
            [-3.670075, 2.360153, 3.635230],
            [-3.177186, 2.294265, 3.775704],
            [-2.196121, -4.598322, 4.479786],
            [-6.234883, -1.944430, 1.663542],
            [-1.292924, -9.295920, 4.094063],
            [-3.210651, -8.533278, 2.802001],
            [-4.068926, -7.993109, 1.925119],
            [0.000000, 6.545390, 5.027311],
            [0.000000, -9.403378, 4.264492],
            [-2.724032, 2.315802, 3.777151],
            [-2.288460, 2.398891, 3.697603],
            [-1.998311, 2.496547, 3.689148],
            [-6.130040, 3.399261, 2.038516],
            [-2.288460, 2.886504, 3.775031],
            [-2.724032, 2.961810, 3.871767],
            [-3.177186, 2.964136, 3.876973],
            [-3.670075, 2.927714, 3.724325],
            [-4.018389, 2.857357, 3.482983],
            [-7.555811, 4.106811, -0.991917],
            [-4.018389, 2.483695, 3.440898],
            [0.000000, -2.521945, 5.932265],
            [-1.776217, -2.683946, 5.213116],
            [-1.222237, -1.182444, 5.952465],
            [-0.731493, -2.536683, 5.815343],
            [0.000000, 3.271027, 5.236015],
            [-4.135272, -6.996638, 2.671970],
            [-3.311811, -7.660815, 3.382963],
            [-1.313701, -8.639995, 4.702456],
            [-5.940524, -6.223629, -0.631468],
            [-1.998311, 2.743838, 3.744030],
            [-0.901447, 1.236992, 5.754256],
            [0.000000, -8.765243, 4.891441],
            [-2.308977, -8.974196, 3.609070],
            [-6.954154, -2.439843, -0.131163],
            [-1.098819, -4.458788, 5.120727],
            [-1.181124, -4.579996, 5.189564],
            [-1.255818, -4.787901, 5.237051],
            [-1.325085, -5.106507, 5.205010],
            [-1.546388, -5.819392, 4.757893],
            [-1.953754, -4.183892, 4.431713],
            [-2.117802, -4.137093, 4.555096],
            [-2.285339, -4.051196, 4.582438],
            [-2.850160, -3.665720, 4.484994],
            [-5.278538, -2.238942, 2.861224],
            [-0.946709, 1.907628, 5.196779],
            [-1.314173, 3.104912, 4.231404],
            [-1.780000, 2.860000, 3.881555],
            [-1.845110, -4.098880, 4.247264],
            [-5.436187, -4.030482, 2.109852],
            [-0.766444, 3.182131, 4.861453],
            [-1.938616, -6.614410, 4.521085],
            [0.000000, 1.059413, 6.774605],
            [-0.516573, 1.583572, 6.148363],
            [0.000000, 1.728369, 6.316750],
            [-1.246815, 0.230297, 5.681036],
            [0.000000, -7.942194, 5.181173],
            [0.000000, -6.991499, 5.153478],
            [-0.997827, -6.930921, 4.979576],
            [-3.288807, -5.382514, 3.795752],
            [-2.311631, -1.566237, 4.590085],
            [-2.680250, -6.111567, 4.096152],
            [-3.832928, -1.537326, 4.137731],
            [-2.961860, -2.274215, 4.440943],
            [-4.386901, -2.683286, 3.643886],
            [-1.217295, -7.834465, 4.969286],
            [-1.542374, -0.136843, 5.201008],
            [-3.878377, -6.041764, 3.311079],
            [-3.084037, -6.809842, 3.814195],
            [-3.747321, -4.503545, 3.726453],
            [-6.094129, -3.205991, 1.473482],
            [-4.588995, -4.728726, 2.983221],
            [-6.583231, -3.941269, 0.070268],
            [-3.492580, -3.195820, 4.130198],
            [-1.255543, 0.802341, 5.307551],
            [-1.126122, -0.933602, 6.538785],
            [-1.443109, -1.142774, 5.905127],
            [-0.923043, -0.529042, 7.003423],
            [-1.755386, 3.529117, 4.327696],
            [-2.632589, 3.713828, 4.364629],
            [-3.388062, 3.721976, 4.309028],
            [-4.075766, 3.675413, 4.076063],
            [-4.622910, 3.474691, 3.646321],
            [-5.171755, 2.535753, 2.670867],
            [-7.297331, 0.763172, -0.048769],
            [-4.706828, 1.651000, 3.109532],
            [-4.071712, 1.476821, 3.476944],
            [-3.269817, 1.470659, 3.731945],
            [-2.527572, 1.617311, 3.865444],
            [-1.970894, 1.858505, 3.961782],
            [-1.579543, 2.097941, 4.084996],
            [-7.664182, 0.673132, -2.435867],
            [-1.397041, -1.340139, 5.630378],
            [-0.884838, 0.658740, 6.233232],
            [-0.767097, -0.968035, 7.077932],
            [-0.460213, -1.334106, 6.787447],
            [-0.748618, -1.067994, 6.798303],
            [-1.236408, -1.585568, 5.480490],
            [-0.387306, -1.409990, 6.957705],
            [-0.319925, -1.607931, 6.508676],
            [-1.639633, 2.556298, 3.863736],
            [-1.255645, 2.467144, 4.203800],
            [-1.031362, 2.382663, 4.615849],
            [-4.253081, 2.772296, 3.315305],
            [-4.530000, 2.910000, 3.339685],
            [0.463928, 0.955357, 6.633583],
            [4.253081, 2.577646, 3.279702],
            [0.416106, -1.466449, 6.447657],
            [7.087960, 5.434801, 0.099620],
            [2.628639, 2.035898, 3.848121],
            [3.198363, 1.985815, 3.796952],
            [3.775151, 2.039402, 3.646194],
            [4.465819, 2.422950, 3.155168],
            [2.164289, 2.189867, 3.851822],
            [3.208229, 3.223926, 4.115822],
            [2.673803, 3.205337, 4.092203],
            [3.745193, 3.165286, 3.972409],
            [4.161018, 3.059069, 3.719554],
            [5.062006, 1.934418, 2.776093],
            [2.266659, -7.425768, 4.389812],
            [4.445859, 2.663991, 3.173422],
            [7.214530, 2.263009, 0.073150],
            [5.799793, 2.349546, 2.204059],
            [2.844939, -0.720868, 4.433130],
            [0.711452, -3.329355, 5.877044],
            [0.606033, -3.924562, 5.444923],
            [1.431615, -3.500953, 5.496189],
            [1.914910, -3.803146, 5.028930],
            [1.131043, -3.973937, 5.189648],
            [1.563548, -4.082763, 4.842263],
            [2.650112, -5.003649, 4.188483],
            [0.427049, -1.094134, 7.360529],
            [0.496396, -0.475659, 7.440358],
            [5.253307, 3.881582, 3.363159],
            [1.718698, 0.974609, 4.558359],
            [1.608635, -0.942516, 5.814193],
            [1.651267, -0.610868, 5.581319],
            [4.765501, -0.701554, 3.534632],
            [0.478306, 0.295766, 7.101013],
            [3.734964, 4.508230, 4.550454],
            [4.588603, 4.302037, 4.048484],
            [6.279331, 6.615427, 1.425850],
            [1.220941, 4.142165, 5.106035],
            [2.193489, 3.100317, 4.000575],
            [3.102642, -4.352984, 4.095905],
            [6.719682, -4.788645, -1.745401],
            [1.193824, -1.306795, 5.737747],
            [0.729766, -1.593712, 5.833208],
            [2.456206, -4.342621, 4.283884],
            [2.204823, -4.304508, 4.162499],
            [4.985894, 4.802461, 3.751977],
            [1.592294, -1.257709, 5.456949],
            [2.644548, 4.524654, 4.921559],
            [2.760292, 5.100971, 5.015990],
            [3.523964, 8.005976, 3.729163],
            [5.599763, 5.715470, 2.724259],
            [3.063932, 6.566144, 4.529981],
            [5.720968, 4.254584, 2.830852],
            [6.374393, 4.785590, 1.591691],
            [0.672728, -3.688016, 5.737804],
            [1.262560, -3.787691, 5.417779],
            [1.732553, -3.952767, 5.000579],
            [1.043625, -1.464973, 5.662455],
            [2.321234, -4.329069, 4.258156],
            [2.056846, -4.477671, 4.520883],
            [2.153084, -4.276322, 4.038093],
            [0.946874, -1.035249, 6.512274],
            [1.469132, -4.036351, 4.604908],
            [1.024340, -3.989851, 4.926693],
            [0.533422, -3.993222, 5.138202],
            [0.769720, -6.095394, 4.985883],
            [0.699606, -5.291850, 5.448304],
            [0.669687, -4.949770, 5.509612],
            [0.630947, -4.695101, 5.449371],
            [0.583218, -4.517982, 5.339869],
            [1.537170, -4.423206, 4.745470],
            [1.615600, -4.475942, 4.813632],
            [1.729053, -4.618680, 4.854463],
            [1.838624, -4.828746, 4.823737],
            [2.368250, -3.106237, 4.868096],
            [7.542244, -1.049282, -2.431321],
            [1.826614, -4.399531, 4.399021],
            [1.929558, -4.411831, 4.497052],
            [0.597442, -2.013686, 5.866456],
            [1.405627, -1.714196, 5.241087],
            [0.662449, -1.819321, 5.863759],
            [2.342340, 0.572222, 4.294303],
            [3.327324, 0.104863, 4.113860],
            [1.726175, -0.919165, 5.273355],
            [5.133204, 7.485602, 2.660442],
            [4.538641, 6.319907, 3.683424],
            [3.986562, 5.109487, 4.466315],
            [2.169681, -5.440433, 4.455874],
            [1.395634, 5.011963, 5.316032],
            [1.619500, 6.599217, 4.921106],
            [1.891399, 8.236377, 4.274997],
            [4.195832, 2.235205, 3.375099],
            [5.733342, 1.411738, 2.431726],
            [1.859887, 2.355757, 3.843181],
            [4.988612, 3.074654, 3.083858],
            [1.303263, 1.416453, 4.831091],
            [1.305757, -0.672779, 6.415959],
            [6.465170, 0.937119, 1.689873],
            [5.258659, 0.945811, 2.974312],
            [4.432338, 0.722096, 3.522615],
            [3.300681, 0.861641, 3.872784],
            [2.430178, 1.131492, 4.039035],
            [1.820731, 1.467954, 4.224124],
            [0.563221, 2.307693, 5.566789],
            [6.338145, -0.529279, 1.881175],
            [5.587698, 3.208071, 2.687839],
            [0.242624, -1.462857, 7.071491],
            [1.611251, 0.339326, 4.895421],
            [7.743095, 2.364999, -2.005167],
            [1.391142, 1.851048, 4.448999],
            [1.785794, -0.978284, 4.850470],
            [4.670959, 2.664461, 3.084075],
            [1.333970, -0.283761, 6.097047],
            [7.270895, -2.890917, -2.252455],
            [1.856432, 2.585245, 3.757904],
            [0.923388, 0.073076, 6.671944],
            [5.000589, -6.135128, 1.892523],
            [5.085276, -7.178590, 0.714711],
            [7.159291, -0.811820, -0.072044],
            [5.843051, -5.248023, 0.924091],
            [6.847258, 3.662916, 0.724695],
            [2.412942, -8.258853, 4.119213],
            [0.179909, -1.689864, 6.573301],
            [2.103655, -0.163946, 4.566119],
            [6.407571, 2.236021, 1.560843],
            [3.670075, 2.360153, 3.635230],
            [3.177186, 2.294265, 3.775704],
            [2.196121, -4.598322, 4.479786],
            [6.234883, -1.944430, 1.663542],
            [1.292924, -9.295920, 4.094063],
            [3.210651, -8.533278, 2.802001],
            [4.068926, -7.993109, 1.925119],
            [2.724032, 2.315802, 3.777151],
            [2.288460, 2.398891, 3.697603],
            [1.998311, 2.496547, 3.689148],
            [6.130040, 3.399261, 2.038516],
            [2.288460, 2.886504, 3.775031],
            [2.724032, 2.961810, 3.871767],
            [3.177186, 2.964136, 3.876973],
            [3.670075, 2.927714, 3.724325],
            [4.018389, 2.857357, 3.482983],
            [7.555811, 4.106811, -0.991917],
            [4.018389, 2.483695, 3.440898],
            [1.776217, -2.683946, 5.213116],
            [1.222237, -1.182444, 5.952465],
            [0.731493, -2.536683, 5.815343],
            [4.135272, -6.996638, 2.671970],
            [3.311811, -7.660815, 3.382963],
            [1.313701, -8.639995, 4.702456],
            [5.940524, -6.223629, -0.631468],
            [1.998311, 2.743838, 3.744030],
            [0.901447, 1.236992, 5.754256],
            [2.308977, -8.974196, 3.609070],
            [6.954154, -2.439843, -0.131163],
            [1.098819, -4.458788, 5.120727],
            [1.181124, -4.579996, 5.189564],
            [1.255818, -4.787901, 5.237051],
            [1.325085, -5.106507, 5.205010],
            [1.546388, -5.819392, 4.757893],
            [1.953754, -4.183892, 4.431713],
            [2.117802, -4.137093, 4.555096],
            [2.285339, -4.051196, 4.582438],
            [2.850160, -3.665720, 4.484994],
            [5.278538, -2.238942, 2.861224],
            [0.946709, 1.907628, 5.196779],
            [1.314173, 3.104912, 4.231404],
            [1.780000, 2.860000, 3.881555],
            [1.845110, -4.098880, 4.247264],
            [5.436187, -4.030482, 2.109852],
            [0.766444, 3.182131, 4.861453],
            [1.938616, -6.614410, 4.521085],
            [0.516573, 1.583572, 6.148363],
            [1.246815, 0.230297, 5.681036],
            [0.997827, -6.930921, 4.979576],
            [3.288807, -5.382514, 3.795752],
            [2.311631, -1.566237, 4.590085],
            [2.680250, -6.111567, 4.096152],
            [3.832928, -1.537326, 4.137731],
            [2.961860, -2.274215, 4.440943],
            [4.386901, -2.683286, 3.643886],
            [1.217295, -7.834465, 4.969286],
            [1.542374, -0.136843, 5.201008],
            [3.878377, -6.041764, 3.311079],
            [3.084037, -6.809842, 3.814195],
            [3.747321, -4.503545, 3.726453],
            [6.094129, -3.205991, 1.473482],
            [4.588995, -4.728726, 2.983221],
            [6.583231, -3.941269, 0.070268],
            [3.492580, -3.195820, 4.130198],
            [1.255543, 0.802341, 5.307551],
            [1.126122, -0.933602, 6.538785],
            [1.443109, -1.142774, 5.905127],
            [0.923043, -0.529042, 7.003423],
            [1.755386, 3.529117, 4.327696],
            [2.632589, 3.713828, 4.364629],
            [3.388062, 3.721976, 4.309028],
            [4.075766, 3.675413, 4.076063],
            [4.622910, 3.474691, 3.646321],
            [5.171755, 2.535753, 2.670867],
            [7.297331, 0.763172, -0.048769],
            [4.706828, 1.651000, 3.109532],
            [4.071712, 1.476821, 3.476944],
            [3.269817, 1.470659, 3.731945],
            [2.527572, 1.617311, 3.865444],
            [1.970894, 1.858505, 3.961782],
            [1.579543, 2.097941, 4.084996],
            [7.664182, 0.673132, -2.435867],
            [1.397041, -1.340139, 5.630378],
            [0.884838, 0.658740, 6.233232],
            [0.767097, -0.968035, 7.077932],
            [0.460213, -1.334106, 6.787447],
            [0.748618, -1.067994, 6.798303],
            [1.236408, -1.585568, 5.480490],
            [0.387306, -1.409990, 6.957705],
            [0.319925, -1.607931, 6.508676],
            [1.639633, 2.556298, 3.863736],
            [1.255645, 2.467144, 4.203800],
            [1.031362, 2.382663, 4.615849],
            [4.253081, 2.772296, 3.315305],
            [4.530000, 2.910000, 3.339685]
        ], dtype=float)
        self.face_model_all -= self.face_model_all[1]
        self.face_model_all *= np.array([1, -1, -1])  # fix axis
        self.face_model_all *= 10
        self.landmarks_ids = [33, 133, 362, 263, 61, 291, 1]  # reye, leye, mouth
        self.face_model = np.asarray([self.face_model_all[i] for i in self.landmarks_ids])
        self.camera_matrix = camera_matrix
        self.dist_coefficients = dist_coefficients

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Model.load_from_checkpoint('./pretrained_gaze_model.ckpt').to(self.device)
        self.model.eval()

    # def trans2onxx(self, path):



    def calibrate_screen(self):

        (width_mm, height_mm), (width_pixels, height_pixels) = get_monitor_dimensions()
        np.save('./mean_std.npy', np.array([self.fix_pos.mean(axis=0), self.fix_pos.std(axis=0)]))
        # 对self.fix_pos按列进行标准化
        self.fix_pos = (self.fix_pos - self.fix_pos.mean(axis=0)) / self.fix_pos.std(axis=0)
        cam_eye_points = []
        pitches, yaws = [], []

        del_idx = []
        for i in range(len(self.frames)):
            frame = self.frames[i]

            pitch_yaw, cam_eye_point, _ = self.process(frame)
            if pitch_yaw is None:
                del_idx.append(i)
                continue

            pitch = pitch_yaw[0]
            yaw = pitch_yaw[1]

            cam_eye_points.append(cam_eye_point.flatten().tolist())
            pitches.append(pitch)
            yaws.append(yaw)

        self.fix_pos = np.delete(self.fix_pos, del_idx, axis=0)
        cam_eye_points = np.array(cam_eye_points)
        model = self.regression
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        model = model.to(self.device)
        input_t = torch.Tensor(
            np.concatenate([cam_eye_points, np.array(pitches).reshape(-1, 1), np.array(yaws).reshape(-1, 1)], axis=1))
        input_t = input_t.to(self.device)

        nums_sequences = 0
        if nums_sequences != 0:
            temp_input_t = input_t.clone()
            input_t = input_t.unsqueeze(1)
            input_t = input_t.repeat(1, nums_sequences, 1)
            for i in range(input_t.shape[0]):
                if i < nums_sequences:
                    input_t[i, :, :] = torch.cat([temp_input_t[:i, :]] + [temp_input_t[i, :].unsqueeze(0) for _ in range(nums_sequences - i)])
                else:
                    input_t[i, :, :] = temp_input_t[i - nums_sequences + 1 : i + 1, :]


        train_size = int(0.8 * len(input_t))
        test_size = len(input_t) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            torch.utils.data.TensorDataset(input_t, torch.Tensor(self.fix_pos)), [train_size, test_size])
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
        full_loader = DataLoader(torch.utils.data.TensorDataset(input_t, torch.Tensor(self.fix_pos)), batch_size=32, shuffle=True)
        best_test_loss = 1e8
        for _ in range(1000):
            losses = []
            for idx, (data, fix_pos) in enumerate(full_loader):
                model.train()
                data = data.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, fix_pos.to(self.device))
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if _ % 100 == 0:
                model.eval()
                test_losses = []
                with torch.no_grad():
                    for idx, (data, fix_pos) in enumerate(test_loader):
                        data = data.to(self.device)
                        output = model(data)
                        test_loss = criterion(output, fix_pos.to(self.device))
                        if test_loss.item() < best_test_loss:
                            torch.save(model.state_dict(), './regression_model.pth')
                        test_losses.append(test_loss.item())
                print('loss: {}, test_loss: {}'.format(np.mean(losses), np.mean(test_losses)))




    def gaze_3dto2d(self, gaze_vector):

        pitch = np.arcsin(-gaze_vector[1])
        yaw = np.arctan2(-gaze_vector[0], -gaze_vector[2])

        return pitch, yaw

    def process(self, frame):

        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            # head pose estimation
            face_landmarks = np.asarray(
                [[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
            face_landmarks = np.asarray([face_landmarks[i] for i in self.landmarks_ids])

            success, rvec, tvec, inliers = cv2.solvePnPRansac(self.face_model, face_landmarks, self.camera_matrix,
                                                              self.dist_coefficients, rvec=None, tvec=None,
                                                              useExtrinsicGuess=True,
                                                              flags=cv2.SOLVEPNP_EPNP)  # Initial fit
            for _ in range(10):
                success, rvec, tvec = cv2.solvePnP(self.face_model, face_landmarks, self.camera_matrix, self.dist_coefficients,
                                                   rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)  # Second fit for higher accuracy

            # data preprocessing # 世界坐标系到相机坐标系
            face_model_transformed, face_model_all_transformed = get_face_landmarks_in_ccs(self.camera_matrix,
                                                                                           self.dist_coefficients,
                                                                                           frame.shape, results,
                                                                                           self.face_model, self.face_model_all,
                                                                                           self.landmarks_ids)
            left_eye_center = 0.5 * (face_model_transformed[:, 2] + face_model_transformed[:, 3]).reshape(
                (3, 1))  # center eye
            right_eye_center = 0.5 * (face_model_transformed[:, 0] + face_model_transformed[:, 1]).reshape(
                (3, 1))  # center eye
            face_center = face_model_transformed.mean(axis=1).reshape((3, 1))

            img_warped_left_eye, _, _ = normalize_single_image(image_rgb, rvec, None, left_eye_center, self.camera_matrix)
            img_warped_right_eye, _, _ = normalize_single_image(image_rgb, rvec, None, right_eye_center, self.camera_matrix)
            img_warped_face, _, rotation_matrix = normalize_single_image(image_rgb, rvec, None, face_center,
                                                                         self.camera_matrix, is_eye=False)

            transform = A.Compose([
                A.Normalize(),
                ToTensorV2()
            ])

            person_idx = torch.Tensor([0]).unsqueeze(0).long().to(
                self.device)  # TODO adapt this depending on the loaded model
            full_face_image = transform(image=img_warped_face)["image"].unsqueeze(0).float().to(self.device)
            left_eye_image = transform(image=img_warped_left_eye)["image"].unsqueeze(0).float().to(self.device)
            right_eye_image = transform(image=img_warped_right_eye)["image"].unsqueeze(0).float().to(self.device)

            # prediction
            output = self.model(person_idx, full_face_image, right_eye_image, left_eye_image).squeeze(
                0).detach().cpu().numpy()

            gaze_vector_3d_normalized = gaze_2d_to_3d(output)
            gaze_vector = np.dot(np.linalg.inv(rotation_matrix), gaze_vector_3d_normalized)

            plane = plane_equation(np.eye(3), np.asarray([[0], [0], [0]]))
            plane_w = plane[0:3]
            plane_b = plane[3]

            result = ray_plane_intersection(face_center.reshape(3), gaze_vector, plane_w, plane_b)

            return self.gaze_3dto2d(gaze_vector), (left_eye_center + right_eye_center) / 2, result

        else:
            return None, None, None

def vector_to_angle(vector: np.ndarray) -> np.ndarray:
    assert vector.shape == (3, )
    x, y, z = vector
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw])



if __name__ == '__main__':

    camera_matrix, dist_coefficients = get_camera_matrix('./calibration_matrix.yaml')

    calibrator = screen_camera_calibration(camera_matrix, dist_coefficients)

    calibrator.calibrate_screen()




