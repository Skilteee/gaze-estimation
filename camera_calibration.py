import glob
from datetime import datetime

import cv2
import numpy as np
import yaml

from webcam import WebcamSource


def record_one_minute_video(width: int, height: int, fps: int = 30):
    # 设置录制时间和总帧数
    recording_time_in_seconds = 20
    total_frames = recording_time_in_seconds * fps

    # 创建 WebcamSource 对象和 VideoWriter 对象
    source = WebcamSource(width=width, height=height, fps=fps, buffer_size=10)
    video_writer = cv2.VideoWriter(
        './docs/video1.mp4',
        cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height)
    )

    # 录制视频
    for idx, frame in enumerate(source):
        video_writer.write(frame)
        source.show(frame, only_print=idx % 5 != 0)

        # 检查是否达到总帧数
        if idx + 1 == total_frames:
            break

    # 释放资源
    video_writer.release()


def calibration(image_path, every_nth: int = 1, debug: bool = False, chessboard_grid_size=(7, 7)):
    """
    Perform camera calibration on the previously collected images.
    Creates `calibration_matrix.yaml` with the camera intrinsic matrix and the distortion coefficients.

    :param image_path: path to all png images
    :param every_nth: only use every n_th image
    :param debug: preview the matched chess patterns
    :param chessboard_grid_size: size of chess pattern
    :return:
    """

    x, y = chessboard_grid_size

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((y * x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(f'{image_path}/*.png')[::every_nth]

    found = 0
    for fname in images:
        img = cv2.imread(fname)  # Capture frame-by-frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (x, y), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            found += 1

            if debug:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, chessboard_grid_size, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(100)

    print("Number of images used for calibration: ", found)

    # When everything done, release the capture
    cv2.destroyAllWindows()

    # calibration
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('rms', rms)

    # transform the matrix and distortion coefficients to writable lists
    data = {
        'rms': np.asarray(rms).tolist(),
        'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist()
    }

    # and save it to a file
    with open("calibration_matrix.yaml", "w") as f:
        yaml.dump(data, f)

    print(data)


def split_video_into_frames(path, output):
    # 读取视频
    cap = cv2.VideoCapture(path)

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 逐帧保存
    for idx in range(total_frames):
        # 读取一帧
        ret, frame = cap.read()

        # 保存一帧
        cv2.imwrite(f'{output}/frame_{idx}.png', frame)

    # 释放资源
    cap.release()



if __name__ == '__main__':
    # 1. record video
    # record_one_minute_video(width=640, height=480)
    # 2. split video into frames
    # split_video_into_frames('./docs/video.mp4', './frames')
    # 3. run calibration on images
    calibration('./frames', 30, debug=True)
