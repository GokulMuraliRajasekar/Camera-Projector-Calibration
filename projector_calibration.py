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
    projector_size = (1280, 800)
    
    # calibration files
    folder = './hi_res/'
    files = os.listdir(folder)
    files = reorder_files(files)
    
    # initialize detection/calibration objects
    detect = detection(chess_board_shape, circle_grid_shape)
    camera = calibration('logitech_1280_720.yaml', chess_board_shape, circle_grid_shape, camera_size)
    projector = calibration(None, chess_board_shape, circle_grid_shape, projector_size)
    
    # detect circle grid in image to be projected
    projector_image = cv2.imread('./calib_pdf_resized.png')
    ret, projector_image_points = detect.find_circle_grid(projector_image)
    
    # read files and detect chessboard corners
    print('\nprocessing images')
    for file in files:
        if 'png' in file:
            img = cv2.imread(folder + file)
            gray = cv2.imread(folder + file, 0)
            
            board_ret, board_corners = detect.find_chess_corner_points(gray)
            circle_ret, circle_points = detect.find_circle_grid(gray)
            
            if board_ret & circle_ret:

                ret, rvec, tvec = cv2.solvePnP(camera.generate_camera_object_points(),
                                           board_corners,
                                           camera.camera_matrix,
                                           camera.distortion_matrix)
                object_points = camera.back_project(circle_points, rvec, tvec)
                projector.add_object_points(object_points)
                projector.add_image_points(projector_image_points) 
                camera.add_image_points(circle_points)
                
    print('\ncalibrating projector')
    projector.calibrate()
    projector.calculate_reprojection_error()
    projector.save_calibration('projector_hi_res.yaml')
    
    # perform stereo calibration
    print('\ncalibrating camera-projector pair')
    ret, cam_mat, cam_dist, proj_mat, proj_dist, proj_R, proj_T, _, _ = cv2.stereoCalibrate(np.float32(projector.object_points),
                                                                                      camera.image_points,
                                                                                      projector.image_points,
                                                                                      camera.camera_matrix,
                                                                                      camera.distortion_matrix,
                                                                                      projector.camera_matrix,
                                                                                      projector.distortion_matrix,
                                                                                      camera.imager_size,
                                                                                      flags = cv2.CALIB_USE_INTRINSIC_GUESS)
    print('camera projector calibration error : ', ret)
    projector.save_stereo_calibration('camera_projector_hi_res.yaml', proj_R, proj_T)
    print('\ncalibration done!')

if __name__ == '__main__':
    
    main()
