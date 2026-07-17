import json

import cv2
import numpy as np


class LaneDetector:

    LEFT = 0
    CENTER = 1
    RIGHT = 2

    def __init__(self):
        with open("assets/configs/lanes.json", "r") as f:
            cfg = json.load(f)
            
        self.perspectives = {}
        for lane_idx, lines in cfg["perspectives"].items():
            self.perspectives[int(lane_idx)] = {
                "L_Left": np.array(lines.get("L_Left", []), dtype=np.float32),
                "L_Right": np.array(lines.get("L_Right", []), dtype=np.float32),
                "C_Left": np.array(lines.get("C_Left", []), dtype=np.float32),
                "C_Right": np.array(lines.get("C_Right", []), dtype=np.float32),
                "R_Left": np.array(lines.get("R_Left", []), dtype=np.float32),
                "R_Right": np.array(lines.get("R_Right", []), dtype=np.float32)
            }
            
    def _get_x_at_y(self, polyline, target_y):
        """Piecewise linear interpolation along a curved polyline to find X for a given Y"""
        if len(polyline) == 0:
            return 0.0
            
        # Sort polyline by Y (top to bottom)
        sorted_line = polyline[polyline[:, 1].argsort()]
        
        if target_y <= sorted_line[0, 1]:
            return sorted_line[0, 0]
        if target_y >= sorted_line[-1, 1]:
            return sorted_line[-1, 0]
            
        for i in range(len(sorted_line) - 1):
            p1, p2 = sorted_line[i], sorted_line[i+1]
            if p1[1] <= target_y <= p2[1]:
                if abs(p2[1] - p1[1]) < 1:
                    return p1[0]
                t = (target_y - p1[1]) / (p2[1] - p1[1])
                return p1[0] + t * (p2[0] - p1[0])
                
        return sorted_line[0, 0]

    def get_object_lane(self, center_x, center_y, agent_lane):
        """Calculates which lane an object is in using the exact traced curves"""
        p = self.perspectives.get(int(agent_lane), self.perspectives[1])
        
        # If the model hasn't been calibrated yet with 6 lines, fallback to center
        if len(p["L_Right"]) == 0:
            return 1
            
        l_right = self._get_x_at_y(p["L_Right"], center_y)
        c_left = self._get_x_at_y(p["C_Left"], center_y)
        c_right = self._get_x_at_y(p["C_Right"], center_y)
        r_left = self._get_x_at_y(p["R_Left"], center_y)
        
        # Split the gap between the rails (if the user drew them in the middle, they overlap anyway)
        left_center_divider = (l_right + c_left) / 2.0
        center_right_divider = (c_right + r_left) / 2.0
        
        if center_x < left_center_divider:
            return 0  # Left
        elif center_x < center_right_divider:
            return 1  # Center
        else:
            return 2  # Right

    def draw(self, frame, agent_lane=1):
        """Draws the precise curved polylines onto the frame for debugging"""
        p = self.perspectives.get(int(agent_lane), self.perspectives[1])
        if len(p["L_Right"]) == 0:
            return frame
            
        colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        lines = [p["L_Left"], p["L_Right"], p["C_Left"], p["C_Right"], p["R_Left"], p["R_Right"]]
        
        for idx, line_pts in enumerate(lines):
            color = colors[idx]
            for i in range(1, len(line_pts)):
                p1 = tuple(line_pts[i-1].astype(int))
                p2 = tuple(line_pts[i].astype(int))
                cv2.line(frame, p1, p2, color, 3)

        return frame