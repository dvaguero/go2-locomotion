import os
import time
import json
import functools
import shutil
from typing import TypeVar, Type
from pathlib import Path 

import numpy as np
import jax
from jax import numpy as jnp
import orbax.checkpoint as orbax
from flax import nnx
import mujoco
from mujoco import viewer
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import checkpoint

from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from mujoco_playground._src.gait import draw_joystick_command

from mujoco_playground._src.locomotion.go2 import go2_constants
from mujoco_playground._src.locomotion.go2.base import get_assets
# from mujoco_playground.experimental.sim2sim.gamepad_reader import Gamepad

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


class PPOController:
   
    def __init__(self, path_to_checkpoint: str,
                 default_angles: np.ndarray,
                 n_substeps: int,
                 action_scale: float = 0.5,
                 vel_scale_x: float = 1.5,
                 vel_scale_y: float = 0.8,
                 vel_scale_rot: float = 2 * np.pi,):
      
        self.policy = jax.jit(checkpoint.load_policy(path_to_checkpoint))
        self._output_names = ["continuous_actions"]

        self._action_scale = action_scale
        self._default_angles = default_angles
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)

        self._counter = 0
        self._n_substeps = n_substeps

        self._optimizer_key = jax.random.key(0)


    def get_joy_cmd(self): #La idea es que jueguen con esta funcion para enviar cmd al robot

        return np.array([1.0, 0.0, 1.0])


    def get_observation(self, model, data):
       
        linvel = data.sensor("local_linvel").data
        gyro = data.sensor("gyro").data
        imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_angles = data.qpos[7:] - self._default_angles
        joint_velocities = data.qvel[6:]
        obs = np.hstack([
            linvel,
            gyro,
            gravity,
            joint_angles,
            joint_velocities,
            self._last_action,
            self.get_joy_cmd() ])
       
        return {"state": obs.astype(np.float32)}
    

    def get_cmd(self, model, data):
       
        key, rng = jax.random.split(self._optimizer_key)
       
        self._counter += 1
        if self._counter % self._n_substeps == 0:
           
            obs = self.get_observation(model, data)
            pred_cmd = self.policy(obs, rng)[0]
            self._last_action = pred_cmd
            data.ctrl[:] = pred_cmd * self._action_scale + self._default_angles



if __name__ == "__main__":

    # Si al cargar la politica ven un error de una llave con vlor None,
    # eliminen en el json del checkpoint la llave "mean_kernel_init_fn"
    # probablemente su version de brax le falta un parche que salio a posteriori.

    ctrl_dt = 0.02
    sim_dt = 0.004
    n_substeps = int(round(ctrl_dt / sim_dt))
    checkpoint_path = Path("/root/EL7009_projects/mujoco_playground_el7009_project/go2_train_logs/exp_20260625_184739_401/000206438400")

    m = mujoco.MjModel.from_xml_path(
        go2_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
        assets=get_assets())
    d = mujoco.MjData(m)
    
    policy = PPOController( path_to_checkpoint=checkpoint_path.as_posix(),
        default_angles=np.array(m.keyframe("home").qpos[7:]),
        n_substeps=n_substeps,
        action_scale=0.5,
        vel_scale_x=1.5,
        vel_scale_y=0.8,
        vel_scale_rot=2 * np.pi)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() : 

            start_step = time.time()            
            policy.get_cmd(m, d)
            mujoco.mj_step(m, d)
            viewer.sync()
            dt = time.time() - start_step

            waiting_time = time_until_next_step = m.opt.timestep - dt
            if time_until_next_step > 0:
                time.sleep(waiting_time)
