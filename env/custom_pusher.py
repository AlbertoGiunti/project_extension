import numpy as np
from gym import utils
import mujoco_py
from gym.envs.mujoco import MujocoEnv
from gym import spaces
from gym.envs.registration import register

class CustomPusher(MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        utils.EzPickle.__init__(self)
        self.model = mujoco_py.load_model_from_path("env/assets/pusher.xml")
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.frame_skip = 5
        
        # Add tracking for current object
        self.current_object_idx = 0
        self.objects = ["object_red", "object_blue", "object_green"]
        self.goals = ["goal_red", "goal_blue", "goal_green"]
        self.object_positions = {
            "object_red": np.array([0.0, -0.1]),
            "object_blue": np.array([0.35, -0.05]),
            "object_green": np.array([0.55, -0.05])
        }
        self.goal_positions = {
            "goal_red": np.array([0.6, -0.2]),
            "goal_blue": np.array([0.0, 0.3]),
            "goal_green": np.array([-0.6, -0.2])
        }
        
    def _hide_object(self, obj_name):
        """Hide an object by moving it far away"""
        self.sim.data.set_joint_qpos(f"{obj_name}_slidex", 999.0)
        self.sim.data.set_joint_qpos(f"{obj_name}_slidey", 999.0)
        
    def _show_object(self, obj_name, position):
        """Show an object at the specified position"""
        self.sim.data.set_joint_qpos(f"{obj_name}_slidex", position[0])
        self.sim.data.set_joint_qpos(f"{obj_name}_slidey", position[1])
        
    def step(self, a):
        current_obj = self.objects[self.current_object_idx]
        current_goal = self.goals[self.current_object_idx]
        
        # Calculate reward only for current object
        vec_1 = self.sim.data.get_body_xpos(current_obj) - self.sim.data.get_body_xpos("tips_arm")
        vec_2 = self.sim.data.get_body_xpos(current_obj) - self.sim.data.get_body_xpos(current_goal)
        
        reward_near = -np.linalg.norm(vec_1)
        reward_dist = -np.linalg.norm(vec_2)
        reward_ctrl = -0.1 * np.square(a).sum()
        
        total_reward = reward_dist + 0.5 * reward_near + reward_ctrl
        
        self.do_simulation(a, self.frame_skip)
        
        # Check if current object reached its goal
        if np.linalg.norm(vec_2) < 0.05:  # Threshold for considering goal reached
            if self.current_object_idx < len(self.objects) - 1:
                # Hide only the current object and show next one
                self._hide_object(current_obj)
                self.current_object_idx += 1
                next_obj = self.objects[self.current_object_idx]
                self._show_object(next_obj, self.object_positions[next_obj])
                total_reward += 10.0  # Bonus reward for completing an object
            else:
                # All objects completed
                total_reward += 50.0  # Bonus reward for completing all objects
                done = True
                return self._get_obs(), total_reward, done, {"reward_ctrl": reward_ctrl}
        
        if self.render_mode == "human":
            self.render()
            
        ob = self._get_obs()
        done = False
        return ob, total_reward, done, {"reward_ctrl": reward_ctrl}
    
    def reset(self):
        self.current_object_idx = 0
        
        # Hide all objects initially
        for obj in self.objects:
            self._hide_object(obj)
            
        # Show all goals at their positions
        for goal, pos in self.goal_positions.items():
            self._show_object(goal, pos)
            
        # Show only the first object
        first_obj = self.objects[0]
        self._show_object(first_obj, self.object_positions[first_obj])
        
        self.sim.forward()
        return self._get_obs()
    
    def _get_obs(self):
        positions = self.sim.data.qpos.flat[:6]
        velocities = self.sim.data.qvel.flat[:6]
        return np.concatenate([positions, velocities])
    
    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0
    
    def render(self, mode="human"):
        if mode == "human":
            if self.viewer is None:
                self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer.render()
        elif mode == "rgb_array":
            return self.sim.render(width=640, height=480, camera_name="track")
        else:
            raise ValueError("Unsupported render mode")
    
    def close(self):
        if self.viewer is not None:
            self.viewer = None

register(
    id="CustomPusher-v0",
    entry_point="%s:CustomPusher" % __name__,
    max_episode_steps=1000,
)
