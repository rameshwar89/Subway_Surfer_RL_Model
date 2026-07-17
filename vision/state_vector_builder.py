import numpy as np
from vision.lane_detector import LaneDetector

class StateVectorBuilder:
    def __init__(self):
        self.lane_detector = LaneDetector()
        # Memory for obstacle debounce: [Left, Center, Right]
        self.last_distances = [1.0, 1.0, 1.0]
        self.last_types = [0.0, 0.0, 0.0]
        self.police_present = 0.0
        # Decay factor: how fast an unseen obstacle drifts away (1.0 = clear)
        # e.g., 0.05 means it moves 5% of the screen per frame away if unseen
        self.decay = 0.05 
        


    def build_vector(self, tracked_entities, agent_lane, frame_shape, class_names):
        """Transforms tracking dictionary into a dense float array (32-bit floats)."""
        # Start by decaying the memory (pushing obstacles further away)
        lane_distances = [min(1.0, d + self.decay) for d in self.last_distances]
        obstacle_types = list(self.last_types)
        
        # If an obstacle drifted all the way to 1.0 (clear), reset its type
        for i in range(3):
            if lane_distances[i] >= 1.0:
                obstacle_types[i] = 0.0
                
        # Reset police present state every frame. It doesn't need to decay.
        police_present = 0.0
        
        y_max = frame_shape[0]

        for ent in tracked_entities:
            # Normalize distance (0.0 to 1.0)
            norm_dist = max(0.0, min(1.0, (y_max - ent["center_y"]) / y_max))
            
            # Map integer class ID to string name
            cls_id = ent["class"]
            cls_name = class_names.get(cls_id, "").lower()
            
            # If we detect police, set present flag
            if "police" in cls_name: 
                police_present = 1.0
                continue
                
            obj_lane = self.lane_detector.get_object_lane(ent["center_x"], ent["center_y"], agent_lane)
            
            if norm_dist < lane_distances[obj_lane]:
                lane_distances[obj_lane] = norm_dist
                
                # Semantic Obstacle Mapping
                if "traffic light" in cls_name or "climber" in cls_name:
                    obstacle_types[obj_lane] = 0.2  # Safe / Idle
                elif cls_name == "blocker":
                    obstacle_types[obj_lane] = 0.3  # Requires JUMP or ROLL (both work)
                elif "roll block" in cls_name:
                    obstacle_types[obj_lane] = 0.4  # Requires JUMP only (roll is blocked)
                elif "jump-barrier" in cls_name:
                    obstacle_types[obj_lane] = 0.6  # Requires ROLL only (jump hits head)
                else:
                    # 'train' or 'block pillar'
                    obstacle_types[obj_lane] = 1.0  # Solid / Requires Lane Change
                    
        # Update memory
        self.last_distances = lane_distances
        self.last_types = obstacle_types
        self.police_present = police_present
                
        # State: [AgentLane, LeftDist, CenterDist, RightDist, LeftType, CenterType, RightType, PolicePresent]
        state_vector = np.array([
            agent_lane / 2.0,
            lane_distances[0], lane_distances[1], lane_distances[2],
            obstacle_types[0], obstacle_types[1], obstacle_types[2],
            police_present
        ], dtype=np.float32)
        
        return state_vector
