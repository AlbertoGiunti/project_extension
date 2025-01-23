import numpy as np
from gym import utils
import os

from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm
from gym import spaces
from gym.envs.registration import register
import random


class CustomPusher(MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.colors = ["red", "blue", "green"]
        self.color_rgba = {
            "red": np.array([1, 0, 0, 1]),
            "blue": np.array([0, 0, 1, 1]),
            "green": np.array([0, 1, 0, 1])
        }
        self.color_goal_map = {
            "red": "goal_red",
            "blue": "goal_blue",
            "green": "goal_green"
        }
        self.object_positions = {
            "object": np.array([0.0, -0.1])
        }
        self.goal_positions = {
            "goal_red": np.array([0.6, -0.2]),
            "goal_blue": np.array([0.0, 0.3]),
            "goal_green": np.array([-0.6, -0.2])
        }
        self.current_color = random.choice(self.colors)  # Sceglie un colore a caso per il primo step


        '''
            L'init del MujocoEnv va fatto dopo aver dichiarato gli attributi nuovi
        '''

        MujocoEnv.__init__(self, frame_skip=5)
        utils.EzPickle.__init__(self)


        self.viewer = None
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)
        #self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(7,), dtype=np.float64)


    def step(self, a):
        current_goal = self.color_goal_map[self.current_color]

        vec_1 = self.sim.data.get_body_xpos("object") - self.sim.data.get_body_xpos("tips_arm")
        vec_2 = self.sim.data.get_body_xpos("object") - self.sim.data.get_body_xpos(current_goal)

        reward_near = -np.linalg.norm(vec_1)
        reward_dist = -np.linalg.norm(vec_2)
        reward_ctrl = -0.1 * np.square(a).sum()

        total_reward = reward_dist + 0.5 * reward_near + reward_ctrl

        self.do_simulation(a, self.frame_skip)

        if self.render_mode is "human":
            self.render()

        ob = self._get_obs()
        done = False
        return ob, total_reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def reset_model(self):
        # Randomly select a color
        self.current_color = random.choice(self.colors)
        selected_position = self.object_positions["object"]

        # Update the object's colors
        self.model.geom_rgba[self.model.geom_name2id("object_sphere")] = self.color_rgba[self.current_color]
        self.model.geom_rgba[self.model.geom_name2id("object_cylinder")] = self.color_rgba[self.current_color]


        # Set the object's position
        self.model.body_pos[self.model.body_name2id("object")] = np.array(
            [selected_position[0], selected_position[1], -0.275])

        # Show all goals at their positions
        for goal, pos in self.goal_positions.items():
            self._show_object(goal, pos)

        self.sim.forward()
        return self._get_obs()

    def _show_object(self, obj_name, position):
        body_id = self.model.body_name2id(obj_name)
        self.model.body_pos[body_id] = np.array([position[0], position[1], -0.3230])

    def _get_obs(self):
        #Mappa il goal in base al colore
        current_goal = self.color_goal_map[self.current_color]
        return np.concatenate(
            [
                self.sim.data.qpos.flat[:7],
                self.sim.data.qvel.flat[:7],
                self.sim.data.get_body_xpos("tips_arm"),
                self.sim.data.get_body_xpos("object"),
                self.sim.data.get_body_xpos(current_goal),
            ]
        )

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    '''
    def render(self, mode="human"):
        if mode == "human":
            if self.viewer is None:
                self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer.render()
        elif mode == "rgb_array":
            return self.sim.render(width=640, height=480, camera_name="track")
        else:
            raise ValueError("Unsupported render mode")
    '''



register(
    id="CustomPusher-v0",
    entry_point="%s:CustomPusher" % __name__,
    max_episode_steps=1000,
)
