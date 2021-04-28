import cv2
import os
import numpy as np

from camera_projector_calibration import calibration
from camera_projector_calibration import detection

def reorder_files(files):
    file_names = [int(x.replace('.png', '')) for x in files if '.png' in x]
    file_names.sort()
    files = [str(x)+'.png' for x in file_names]
    return files

def main():
    # pattern params
    chess_board_shape = (8, 5)
    circle_grid_shape = (5, 4)
    camera_size = (1280, 720)
    
    # calibration files
    folder = './hi_res/'
    files = os.listdir(folder)
    files = reorder_files(files)
    
    # initialize detection/calibration objects
    detect = detection(chess_board_shape)
    camera = calibration(None, chess_board_shape, None, camera_size)
    
    # read files and detect chessboard corners
    print('reading images')
    for file in files:
        if 'png' in file:
            img = cv2.imread(folder + file)
            gray = cv2.imread(folder + file, 0)

            board_ret, board_corners = detect.find_chess_corner_points(gray)

            if board_ret:
                camera.add_image_points(board_corners)
                camera.add_object_points()
    
    print('starting calibration')
    camera.calibrate()
    camera.calculate_reprojection_error()
    camera.save_calibration('./logitech_1280_720.yaml')

if __name__ == '__main__':
    
    main()
