import os
import time
from datetime import datetime
import functools

import numpy as np
import mediapy as media
import matplotlib
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx
import imageio

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
import jax
from jax import numpy as jp
from ml_collections import config_dict
from orbax import checkpoint as ocp

from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] 
exp_name = f"exp_{timestamp}"

current_dir = os.getcwd()
logs_folder = os.path.join(current_dir, "go2_train_logs")
if not os.path.isdir(logs_folder):
    os.mkdir(logs_folder)

exp_folder = os.path.join(logs_folder, exp_name)
if not os.path.isdir(exp_folder):
    os.mkdir(exp_folder)


xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

matplotlib.use('TkAgg')
plt.ion()
np.set_printoptions(precision=3, suppress=True, linewidth=100)

ENVS = registry.locomotion.ALL_ENVS
print(f"Avaliable envs:\n{ENVS}")
env_name = 'Go2JoystickFlatTerrain'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)
env_ppo_params = locomotion_params.brax_ppo_config(env_name)
registry.get_domain_randomizer(env_name)
print(f"PPO params for Go2 training:\n{env_ppo_params}")

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

def progress(num_steps, metrics):
  #clear_output(wait=True)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  plt.clf()
  plt.xlim([0, env_ppo_params["num_timesteps"] * 1.25])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

  #display(plt.gcf())
  plt.draw()
  plt.gcf().canvas.draw()  # Dibujar explícitamente
  plt.gcf().canvas.flush_events()
  plt.pause(0.01)
  plt.savefig(f'{exp_folder}/reward_{exp_name}.png')


randomizer = registry.get_domain_randomizer(env_name)
ppo_training_params = dict(env_ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in env_ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **env_ppo_params.network_factory
  )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress,
    save_checkpoint_path=exp_folder
)

print("---  Almost ready to training  ---")

make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=registry.load(env_name, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")
plt.show()

# Enable perturbation in the eval env.
print("---  evaluating policy  ---")
env_cfg = registry.get_default_config(env_name)
env_cfg.pert_config.enable = True
env_cfg.pert_config.velocity_kick = [3.0, 6.0]
env_cfg.pert_config.kick_wait_times = [5.0, 15.0]
env_cfg.command_config.a = [1.5, 0.8, 2*jp.pi]
eval_env = registry.load(env_name, config=env_cfg)
velocity_kick_range = [0.0, 0.0]
kick_duration_range = [0.05, 0.2]

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

#make video
from mujoco_playground._src.gait import draw_joystick_command
#Esta velcidad pueden cambiarla a lo que quiera, es para generar un video corto de como funciona la politca
x_vel = 0.0 
y_vel = 0.0
yaw_vel = 3.14


def sample_pert(rng):
  rng, key1, key2 = jax.random.split(rng, 3)
  pert_mag = jax.random.uniform(
      key1, minval=velocity_kick_range[0], maxval=velocity_kick_range[1]
  )
  duration_seconds = jax.random.uniform(
      key2, minval=kick_duration_range[0], maxval=kick_duration_range[1]
  )
  duration_steps = jp.round(duration_seconds / eval_env.dt).astype(jp.int32)
  state.info["pert_mag"] = pert_mag
  state.info["pert_duration"] = duration_steps
  state.info["pert_duration_seconds"] = duration_seconds
  return rng


rng = jax.random.PRNGKey(0)
rollout = []
modify_scene_fns = []

swing_peak = []
rewards = []
linvel = []
angvel = []
track = []
foot_vel = []
rews = []
contact = []
command = jp.array([x_vel, y_vel, yaw_vel])

state = jit_reset(rng)
if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
  rng = sample_pert(rng)
state.info["command"] = command
for i in range(env_cfg.episode_length):
  if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
    rng = sample_pert(rng)
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)
  state.info["command"] = command
  rews.append(
      {k: v for k, v in state.metrics.items() if k.startswith("reward/")}
  )
  rollout.append(state)
  swing_peak.append(state.info["swing_peak"])
  rewards.append(
      {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
  )
  linvel.append(env.get_global_linvel(state.data))
  angvel.append(env.get_gyro(state.data))
  track.append(
      env._reward_tracking_lin_vel(
          state.info["command"], env.get_local_linvel(state.data)
      )
  )

  feet_vel = state.data.sensordata[env._foot_linvel_sensor_adr]
  vel_xy = feet_vel[..., :2]
  vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
  foot_vel.append(vel_norm)

  contact.append(state.info["last_contact"])

  xyz = np.array(state.data.xpos[env._torso_body_id])
  xyz += np.array([0, 0, 0.2])
  x_axis = state.data.xmat[env._torso_body_id, 0]
  yaw = -np.arctan2(x_axis[1], x_axis[0])
  modify_scene_fns.append(
      functools.partial(
          draw_joystick_command,
          cmd=state.info["command"],
          xyz=xyz,
          theta=yaw,
          scl=abs(state.info["command"][0])
          / env_cfg.command_config.a[0],
      )
  )


render_every = 2
fps = 1.0 / eval_env.dt / render_every
traj = rollout[::render_every]
mod_fns = modify_scene_fns[::render_every]

scene_option = mujoco.MjvOption()
scene_option.geomgroup[2] = True
scene_option.geomgroup[3] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

frames = eval_env.render(
    traj,
    camera="track",
    scene_option=scene_option,
    width=640,
    height=480,
    modify_scene_fns=mod_fns,
)
media.show_video(frames, fps=fps, loop=False)

#save the video
print("---  Saving video ---")
output_path = f"{exp_folder}/policy_test.mp4"
# imageio puede escribir directamente
with imageio.get_writer(output_path, fps=fps, format='FFMPEG', codec='libx264') as writer:
    for frame in frames:
        writer.append_data(frame)

print(f"Video guardado en: {output_path}")
print("=="*10)