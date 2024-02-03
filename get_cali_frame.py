import cv2
import numpy as np
import threading
import time
import os
import ctypes

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

drawing = False  # 是否正在绘制
ix, iy = -1, -1  # 起始点坐标
ball_r = 10
idx = 0
local_time = time.strftime("%m-%d-%H-%M", time.localtime())
os.mkdir('./data/' + local_time)
path = './data/' + local_time
cap = cv2.VideoCapture(0)
all_positions = []

# 回调函数，用于处理鼠标事件
def draw_line(event, x, y, flags, param):
    global drawing, ix, iy, img, positions

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (0, 255, 0), 2)

        positions = position_clean(positions)

        # 通过线程绘制圆
        thread = threading.Thread(target=draw_circles, args=(positions,))
        thread.start()
        all_positions.extend(positions)

        positions = []

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (ix, iy), (x, y), (0, 255, 0), 2)
            ix, iy = x, y
            positions.append((x, y))

# 用于绘制圆的函数
def draw_circles(positions):
    global img, idx, f
    img_ = img
    for ball_position in positions:
        # 画白色的空心圆
        cv2.circle(img, ball_position, ball_r, (255, 255, 255), 1)
        ret, frame = cap.read()
        time.sleep(0.2)
        cv2.imwrite('{}/{}.jpg'.format(path, idx), frame)
        idx += 1
        cv2.imshow('image', img)
    time.sleep(1)
    img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

# 计算position中相邻坐标之间的距离, 如果距离小于平均距离, 则删除前一个坐标
def position_clean(positions):
    positions_np = np.array(positions)
    # 计算相邻坐标之间的距离
    dis = [np.linalg.norm(positions_np[i + 1] - positions_np[i]) for i in range(positions_np.shape[0] - 1)]
    sum1 = -1
    idxs = []
    for i in range(len(dis)):
        if sum1 - dis[i] < 0:
            idxs.append(i)
            sum1 = 100
        sum1 -= dis[i]

    positions = [positions[idx] for idx in idxs]

    return positions

(width_mm, height_mm), (screen_width, screen_height) = get_monitor_dimensions()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)


cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_line)
positions = []


while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # 按 ESC 键退出
        with open('{}/screen_position.txt'.format(path), 'a') as f:
            for ball_position in all_positions:
                f.write(str(ball_position[0]) + ' ' + str(ball_position[1]) + '\n')
        break

cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import ctypes
# import time
# import os
#
#
# def get_monitor_dimensions():
#     user32 = ctypes.windll.user32
#
#     # 获取屏幕尺寸
#     width_pixels = user32.GetSystemMetrics(0)  # 0 表示屏幕宽度
#     height_pixels = user32.GetSystemMetrics(1)  # 1 表示屏幕高度
#
#     hdc = user32.GetDC(0)
#
#     # 尝试使用 GetDpiForWindow 获取 DPI
#     hwnd = user32.GetDesktopWindow()
#     dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
#
#     # 如果 GetDpiForWindow 不可用，尝试使用 GetDeviceCaps 获取 DPI
#     if dpi == 0:
#         dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # 88 表示 LOGPIXELSX，表示水平像素数
#
#     width_mm = width_pixels / (dpi / 25.4)  # 计算宽度（毫米）
#     height_mm = height_pixels / (dpi / 25.4)  # 计算高度（毫米）
#
#     user32.ReleaseDC(0, hdc)
#
#     return (width_mm, height_mm), (width_pixels, height_pixels)
#
# # 屏幕大小
# (width_mm, height_mm), (screen_width, screen_height) = get_monitor_dimensions()
# # 在./data下创建一个文件夹名为当前时间的文件夹
# local_time = time.strftime("%m-%d-%H-%M", time.localtime())
# os.mkdir('./data/' + local_time)
# path = './data/' + local_time
# f = open('{}/screen_position.txt'.format(path), 'w')
#
# # 创建黑色背景
# background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
#
# num_cuts = 16
# widths = [int(idx * screen_width // np.sqrt(num_cuts)) for idx in range(1, int(np.sqrt(num_cuts)))]
# heights = [int(idx * screen_height // np.sqrt(num_cuts)) for idx in range(1, int(np.sqrt(num_cuts)))]
#
#
# start = time.time()
#
# # 创建窗口并显示
# cv2.namedWindow('Moving Lines', cv2.WINDOW_NORMAL)
#
# # 窗口全屏
# cv2.setWindowProperty('Moving Lines', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# # 创建窗口并显示
# cv2.namedWindow('Moving Lines', cv2.WINDOW_NORMAL)
#
# # 打开摄像头
# cap = cv2.VideoCapture(0)
#
# line_start = (1, 1)
# line_color = (255, 255, 255)
# line_length = 30
# line_angle = 0
# y_cut = 8
# y_go = screen_height // y_cut
# line_end = (line_start[0] + line_length, line_start[1])
# i = 0

# def update(line_start, flag):
#
#     x = line_start[0]
#     y = line_start[1]
#
#     if x > screen_width or x < 0:
#         y += y_go
#         if flag == 0:
#             x = screen_width
#             flag = 1
#         else:
#             x = 0
#             flag = 0
#         time.sleep(0.5)
#     else:
#         if flag == 0:
#             x += line_length
#         else:
#             x -= line_length
#
#     return (x, y), flag
#
# flag = 0
# while True:
#
#     if line_start[1] >= screen_height:
#         break
#     ret, frame = cap.read()
#     frame1 = cv2.resize(frame, (400, 300))
#     cv2.imshow('frame', frame1)
#
#     background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
#
#     cv2.line(background, line_start, line_end, line_color, 2)
#
#     line_start, flag = update(line_start, flag)
#     line_end = (line_start[0] + line_length, line_start[1])
#
#     # if time.time() - start >= 0.33:
#     #     # 从摄像头读取图像
#     #     cv2.imwrite('{}/{}.jpg'.format(path, i), frame)
#     #     i += 1
#     #     start = time.time()
#     #     f.write(str(line_end[0]) + ' ' + str(line_end[1]) + '\n')
#     # 显示窗口
#     cv2.imshow('Moving Lines', background)
#
#     # 检查按键事件，如果按下ESC键则退出循环
#     key = cv2.waitKey(line_length)
#     if key == 27:
#         break

# def random_line(width0, width1, height0, height1, i):
#     # 线的起始坐标、颜色和终点坐标
#     line_start = (1, 1)
#     line_color = (255, 255, 255)
#     line_length = 30
#     line_angle = 0
#     start = time.time()
#     while True:
#
#         ret, frame = cap.read()
#         frame1 = cv2.resize(frame, (400, 300))
#         cv2.imshow('frame', frame1)
#
#         background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
#
#         # 计算线的终点坐标
#         line_end = (
#             line_start[0] + int(line_length * np.cos(line_angle)),
#             line_start[1] + int(line_length * np.sin(line_angle))
#         )
#         cv2.line(background, (width0, height1), (width1,height1), line_color, 2)
#         cv2.line(background, (width1, height0), (width1,height1), line_color, 2)
#         # 检查线的新坐标是否超出屏幕边界，并进行调整
#         line_end = (max(width0, min(width1, line_end[0])), max(height0, min(height1, line_end[1])))
#
#         # 绘制线
#         cv2.line(background, line_start, line_end, line_color, 2)
#
#         # 更新线的起始坐标
#         line_start = line_end
#
#         if time.time() - start >= 0.33:
#             # 从摄像头读取图像
#             cv2.imwrite('./data/{}/{}.jpg'.format(local_time,i), frame)
#             i += 1
#             start = time.time()
#             f.write(str(line_end[0]) + ' ' + str(line_end[1]) + '\n')
#
#         # 当线段触碰到屏幕边界时设置线的角度为一个随机值
#         if line_start[0] == width0 or line_start[0] == width1 or line_start[1] == height0 or line_start[1] == height1:
#             line_angle = np.random.uniform(0, 2 * np.pi)
#
#         # 随机改变线的角度
#         line_angle += np.random.uniform(-0.3, 0.3)
#
#         # 显示窗口
#         cv2.imshow('Moving Lines', background)
#
#         # 检查按键事件，如果按下ESC键则退出循环
#         key = cv2.waitKey(10)
#         if key == 27:
#             return i
# idx = 0
# num_cuts = 4
# widths = [int(idx * screen_width // np.sqrt(num_cuts)) for idx in range(int(np.sqrt(num_cuts)) + 1)]
# heights = [int(idx * screen_height // np.sqrt(num_cuts)) for idx in range(int(np.sqrt(num_cuts)) + 1)]
#
# for i in range(len(widths) - 1):
#     for j in range(len(heights) - 1):
#         width0, width1 = widths[i], widths[i + 1]
#         height0, height1 = heights[j], heights[j + 1]
#         idx = random_line(width0, width1, height0, height1, idx)




