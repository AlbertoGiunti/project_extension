import numpy as np
from gym import utils
import mujoco_py
from gym.envs.mujoco import MujocoEnv
from gym import spaces
from gym.envs.registration import register


class CustomPusher(MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        utils.EzPickle.__init__(self)
        # Inizializzazione dell'ambiente
        self.model = mujoco_py.load_model_from_path("env/assets/pusher.xml")
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None
        # Definisci lo spazio di osservazione (posizioni e velocità degli oggetti)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        # Definisci lo spazio delle azioni (ad esempio, accelerazioni o forze applicate sull'arm)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)  # esempio con 4 azioni

        self.render_mode = render_mode
        self.frame_skip = 5

    def step(self, a):
        # Calcolo delle distanze e ricompense per ogni oggetto
        rewards = []
        for obj, goal in [("object", "goal"), ("object_blue", "goal_blue"), ("object_green", "goal_green")]:
            vec_1 = self.sim.data.get_body_xpos(obj) - self.sim.data.get_body_xpos("tips_arm")
            vec_2 = self.sim.data.get_body_xpos(obj) - self.sim.data.get_body_xpos(goal)

            reward_near = -np.linalg.norm(vec_1)  # Distanza tra punta e oggetto
            reward_dist = -np.linalg.norm(vec_2)  # Distanza tra oggetto e obiettivo
            rewards.append(reward_dist + 0.5 * reward_near)

        reward_ctrl = -0.1 * np.square(a).sum()  # Penalità per il controllo
        total_reward = sum(rewards) + reward_ctrl

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        return ob, total_reward, False, {"reward_ctrl": reward_ctrl}

    def reset(self):

        # Define initial positions for the objects and goal
        object_pos = np.array([0.0, -0.1])
        object_blue_pos = np.array([0.35, -0.05])
        object_green_pos = np.array([0.55, -0.05])
        goal_pos = np.array([0.6, -0.2])
        goal_blue_pos = np.array([0.0, 0.3])
        goal_green_pos = np.array([-0.6, -0.2])

        '''
        # Optionally add randomness to positions
        object_pos += np.random.uniform(-0.05, 0.05, size=2)
        object_blue_pos += np.random.uniform(-0.05, 0.05, size=2)
        object_green_pos += np.random.uniform(-0.05, 0.05, size=2)
        goal_pos += np.random.uniform(-0.1, 0.1, size=2)
        goal_blue_pos += np.random.uniform(-0.1, 0.1, size=2)
        goal_green_pos += np.random.uniform(-0.1, 0.1, size=2)
        '''
        # Set joint positions in simulation
        self.sim.data.set_joint_qpos("obj_slidex", object_pos[0])
        self.sim.data.set_joint_qpos("obj_slidey", object_pos[1])
        self.sim.data.set_joint_qpos("obj_blue_slidex", object_blue_pos[0])
        self.sim.data.set_joint_qpos("obj_blue_slidey", object_blue_pos[1])
        self.sim.data.set_joint_qpos("obj_green_slidex", object_green_pos[0])
        self.sim.data.set_joint_qpos("obj_green_slidey", object_green_pos[1])
        self.sim.data.set_joint_qpos("goal_slidex", goal_pos[0])
        self.sim.data.set_joint_qpos("goal_slidey", goal_pos[1])
        self.sim.data.set_joint_qpos("goal_blue_slidey", goal_blue_pos[0])
        self.sim.data.set_joint_qpos("goal_blue_slidey", goal_blue_pos[1])
        self.sim.data.set_joint_qpos("goal_green_slidex", goal_green_pos[0])
        self.sim.data.set_joint_qpos("goal_green_slidey", goal_green_pos[1])

        # Reset the simulation
        self.sim.forward()

        # Return the initial observation
        return self._get_obs()

    def _get_obs(self):
        # Collect positions (qpos) and velocities (qvel)
        positions = self.sim.data.qpos.flat[:6]  # Adjust if the simulation includes more joints
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

    def _check_done(self):
        for obj_color, goal_position in self.goal_positions.items():
            obj_position = self.sim.data.get_body_xpos(f'object_{obj_color}')
            if np.linalg.norm(obj_position[:2] - goal_position[:2]) >= 0.05:  # If any object is far
                return False
        return True

register(
        id="CustomPusher-v0",
        entry_point="%s:CustomPusher" % __name__,
        max_episode_steps=1000,
)
