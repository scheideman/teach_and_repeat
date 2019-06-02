#!/usr/bin/env python
import unittest
import numpy as np
from repeat_trajectory import FollowTrajectoryController


class TestFollowTracjectoryController(unittest.TestCase):
    def setUp(self):
        self.controller = FollowTrajectoryController(debug_plot=False)
        self.start_xy = np.array([0, 0])
        self.next_xy = np.array([1, 1])

    def test_cross_track_error_is_correct_sign_when_robot_off_path(self):
        robot_xy = np.array([0, 0.5])
        cte = self.controller._calc_segmented_cte(
            self.start_xy, self.next_xy, robot_xy)
        self.assertGreater(cte, 0)

        robot_xy = np.array([1.0, 0.5])
        cte = self.controller._calc_segmented_cte(
            self.start_xy, self.next_xy, robot_xy)
        self.assertLess(cte, 0)

    def test_check_segment_completion(self):
        robot_xy = np.array([0.5, 0.5])
        done = self.controller._check_segment_done(
            self.start_xy, self.next_xy, robot_xy)

        self.assertFalse(done)

        robot_xy = np.array([3.5, 1])
        done = self.controller._check_segment_done(
            self.start_xy, self.next_xy, robot_xy)

        self.assertTrue(done)

        robot_xy = np.array([-1, -1])
        done = self.controller._check_segment_done(
            self.start_xy, self.next_xy, robot_xy)

        self.assertFalse(done)

    def test_heading_error(self):
        robot_heading = np.pi/4
        heading_error = self.controller._calc_heading_error(
            self.start_xy, self.next_xy, robot_heading)

        self.assertEqual(heading_error, 0)

        robot_heading = 0
        heading_error = self.controller._calc_heading_error(
            self.start_xy, self.next_xy, robot_heading)
        self.assertGreater(heading_error, 0)

        robot_heading = 7 * np.pi/4
        heading_error = self.controller._calc_heading_error(
            self.start_xy, self.next_xy, robot_heading)
        self.assertGreater(heading_error, 0)

        robot_heading = np.pi/2
        heading_error = self.controller._calc_heading_error(
            self.start_xy, self.next_xy, robot_heading)
        self.assertLess(heading_error, 0)

        robot_heading = 5 * np.pi / 4 - 5 * np.pi/180
        heading_error = self.controller._calc_heading_error(
            self.start_xy, self.next_xy, robot_heading)
        self.assertLess(heading_error, 0)

        robot_heading = 5 * np.pi / 4 + 5 * np.pi/180
        heading_error = self.controller._calc_heading_error(
            self.start_xy, self.next_xy, robot_heading)
        self.assertGreater(heading_error, 0)


if __name__ == '__main__':
    import rosunit
    rosunit.rosrun('repeat_trajectory', 'test_follow_trajectory',
                   TestFollowTrajectoryController)
