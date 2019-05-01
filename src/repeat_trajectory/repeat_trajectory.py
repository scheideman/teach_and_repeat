import rospy
import numpy as np
from sensor_msgs.msg import Joy
from std_srvs.srv import Empty 
from enum import Enum
import pyqtgraph as pg
from angles import shortest_angular_distance 

class FollowTrajectoryController(object):
    def __init__(self):
        self._traj = []
        self.initialized = False
        self._cur_ind = 0
        self._next_ind = 1

        self._p = 3.0
        self._d = 0.0
        self._last_cte = 0
        self._max_ind = 0
        self._linear_speed = 0.25
        self._cur_segment_curve = None
        self._robots_progress_curve = None
        self._debug_plot = True
        self._plot_initialized = False
    
    def init_controller(self, pose_array):
        self._traj = self._pose_array_to_xy_array(pose_array) 
        self._max_ind = len(self._traj) - 1
        self.initialized = True
    
    def _pose_array_to_xy_array(self,pose_array):
        xs = []
        ys = []
        for p in pose_array:
            xs.append(p[0,3])
            ys.append(p[1,3])
        traj = np.array(zip(xs,ys))
        return traj 
    
    def get_angular_control(self, robot_xyyaw):
        robot_xy = np.array([robot_xyyaw[0], robot_xyyaw[1]])
        robot_heading = robot_xyyaw[2]
        cte = self._calc_segmented_cte(robot_xy)
        heading_error = self._calc_heading_error(robot_heading)
        ang_speed = self._calc_pd_control(cte, heading_error)
        print("heading: ",heading_error, " cte: ", cte)
        self._last_cte = cte
        return ang_speed

    def get_linear_control(self):
        return self._linear_speed

    def _calc_segmented_cte(self, robot_xy):
        if self._debug_plot:
            self._update_plot(robot_xy)
        
        start_xy, next_xy = self._get_current_segment()

        segment_delta = next_xy - start_xy
        robot_delta = robot_xy - start_xy 

        # project robot_delta onto segment_delta to get distance along segment 
        distance_along_segment = robot_delta.dot(segment_delta) / segment_delta.dot(segment_delta)

        if distance_along_segment >= 1:
            self._update_traj_inds()

        # calculate cross track error, signed vector rejection
        cte = (robot_delta[1] * segment_delta[0] - robot_delta[0]
               * segment_delta[1]) / np.sqrt(segment_delta.dot(segment_delta))

        return cte
    
    def _get_current_segment(self):
        start_xy = self._traj[self._cur_ind,:]
        next_xy = self._traj[self._next_ind,:]
        return (start_xy,next_xy)

    
    def _calc_heading_error(self, robot_heading):
        start_xy, next_xy = self._get_current_segment()
        path_delta = next_xy - start_xy
        path_heading = np.arctan2(path_delta[1],path_delta[0])

        return shortest_angular_distance(robot_heading,path_heading)

    def _update_traj_inds(self):
        self._cur_ind = self._next_ind
        if self._cur_ind == self._max_ind:
            self._next_ind = 0
        else:
            self._next_ind = self._cur_ind + 1
    
    def _calc_pd_control(self, cte, heading_error):
        return heading_error - self._p*cte - self._d * (cte-self._last_cte)

    def _update_plot(self,robot_xy):
        if not self._plot_initialized:
            print('Plot not initialized...')
            return
        pg.QtGui.QApplication.processEvents()
        self._cur_segment_curve.setData(self._traj[self._cur_ind:self._next_ind+1, 0], self._traj[self._cur_ind:self._next_ind+1, 1])
        self._robots_progress_curve.setData([self._traj[self._cur_ind,0], robot_xy[0]], [self._traj[self._cur_ind,1], robot_xy[1]])
    
    def _init_plotting(self):
        self._plot_initialized = True

        self._win = pg.GraphicsWindow()
        self._win.resize(800, 800)
        self._plot = self._win.addPlot()
        # full path
        self._plot.plot(self._traj[:,0], self._traj[:,1], pen = "w")
        # current segment
        self._cur_segment_curve = self._plot.plot(self._traj[self._cur_ind:self._next_ind+1,0],self._traj[self._cur_ind:self._next_ind+1,1],pen="r")
        # robots position relative to current segment
        self._robots_progress_curve = self._plot.plot(self._traj[self._cur_ind:self._next_ind+1,0], self._traj[self._cur_ind:self._next_ind+1,1],pen="g")
