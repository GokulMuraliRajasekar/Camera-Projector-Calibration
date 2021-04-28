import cv2
import numpy as np
import random

class detection():
    
    chess_board_criteria = None
    chess_board_shape = None
    circle_grid_shape = None
    
    camera_image_points = []
    Projector_image_points = []
    object_points = []
    
    def __init__(self,
                 chess_board_shape = None,
                 circle_grid_shape = None,
                 criteria = None):
        self.chess_board_shape = chess_board_shape
        self.circle_grid_shape = circle_grid_shape
        self.chess_board_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        print('detection parameters set')
    
    def find_chess_corner_points(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        ret, corners = cv2.findChessboardCorners(img, (8,5),None)
        if ret == True:
            corners = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), self.chess_board_criteria)
        return ret, corners
    
    def find_circle_grid(self, img):
        ret, circles = cv2.findCirclesGrid(img, 
                                            self.circle_grid_shape, 
                                            flags=cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
        if ret : return ret, circles[::-1]
        return ret, circles
    
    def draw_circles(self, img, corner_pts, bgr = [0, 255, 0]):
        for xy in np.array(corner_pts).reshape(-1, 2):
            x, y = np.int32(xy[0]), np.int32(xy[1])
            cv2.circle(img, (x, y), 2, bgr, 2)
        return img
    
class calibration:
    
    chess_board_shape = None
    circle_grid_shape = None
    camera_matrix = None
    distortion_matrix = None
    imager_size = None
    image_points = []
    object_points = []
    rvecs = []
    tvecs = []
    
    def __init__(self, 
                 calibration_file = None, 
                 chess_board_shape = None, 
                 circle_grid_shape = None,
                 imager_size = None):
        if calibration_file is not None:
            self.load_calibration_from_file(calibration_file)
        else:
            self.camera_matrix = None
            self.distortion_matrix = None
        self.chess_board_shape = chess_board_shape
        self.circle_grid_shape = circle_grid_shape
        self.imager_size = imager_size
        self.object_points = []
        self.image_points = []
    
    def load_calibration_from_file(self, file_name):
        calibration_camera = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
        self.camera_matrix = calibration_camera.getNode('camera_matrix').mat()
        self.distortion_matrix = calibration_camera.getNode('distortion_matrix').mat()
        #return mtx, dst
    
    def save_calibration(self, file_name):
        if not '.yaml' in file_name:
            file_name = file_name + '.yaml'
        cal_file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
        cal_file.write('camera_matrix', self.camera_matrix)
        cal_file.write('distortion_matrix', self.distortion_matrix)
        cal_file.release()
        print('calibration file saved as ', file_name)
       
    def save_stereo_calibration(self, file_name, rvec, tvec):
        if not '.yaml' in file_name:
            file_name = file_name + '.yaml'
        cal_file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
        cal_file.write('rotation_matrix', rvec)
        cal_file.write('translation_vector', tvec)
        cal_file.release()
        print('calibration file saved as ', file_name)
    
    def clear_points(self):
        self.image_points = []
        self.object_points = []
        
    def calibrate(self):
        out = cv2.calibrateCamera(np.float32(self.object_points), 
                                self.image_points, 
                                self.imager_size,
                                self.camera_matrix, 
                                self.distortion_matrix)
        self.camera_matrix = out[1]
        self.distortion_matrix = out[2]
        self.rvecs = out[3]
        self.tvecs = out[4]
        print('calibration done')
    
    def add_image_points(self, image_points):
        self.image_points.append(image_points)
    
    def add_object_points(self, pts = None):
        if pts is None:
            self.object_points.append(self.generate_camera_object_points())
            return
        self.object_points.append(pts)
    
    def generate_camera_object_points(self):
        h, w = self.chess_board_shape
        objp = np.zeros((h * w, 3), np.float32)
        objp[:,:2] = np.mgrid[0:h, 0:w].T.reshape(-1,2)
        return objp
    
    def calculate_reprojection_error(self):
        mean_error = 0
        for i in range(len(self.object_points)):
            imgpoints2, _ = cv2.projectPoints(np.float32(self.object_points[i]),
                                              self.rvecs[i],
                                              self.tvecs[i],
                                              self.camera_matrix,
                                              self.distortion_matrix)
            error = cv2.norm(self.image_points[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        print('reprojection error : ', round(mean_error/len(self.object_points), 2))
    
    def back_project(self, imgpts_h, rvec, tvec): #, mtx, dst):    
        rot3x3 = cv2.Rodrigues(rvec)[0]
        mtx_inv = np.linalg.inv(self.camera_matrix)
        transPlaneToCam = np.dot(np.linalg.inv(rot3x3), tvec)
        total_pts = imgpts_h.shape[0]
        imgpts_h = np.array(imgpts_h.reshape(-1, 2))
        imgpts_h = np.append(imgpts_h, [[1.0] for x in range(total_pts)], axis = 1)
        world_pts = []
        for i in range(imgpts_h.shape[0]):
            col = imgpts_h[i].reshape(3, 1)
            worldPtCam = np.dot(mtx_inv, col)
            worldPtPlane = np.dot(np.linalg.inv(rot3x3), worldPtCam)

            scale = transPlaneToCam[2]/worldPtPlane[2]
            worldPtPlaneReproject = scale * worldPtPlane - transPlaneToCam
            world_pts.append(worldPtPlaneReproject.reshape(1, 3))

        world_pts = np.array(world_pts).reshape(total_pts, 3)
        return world_pts
