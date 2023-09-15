"""
Wraps the MultiClipMocapTracking dm_env into a Gym environment.
"""
import numpy as np
from gym import spaces
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.tasks.reference_pose import types
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.reference_pose import tracking_continuation
from dm_control.locomotion.walkers import cmu_humanoid

import torch

from mocapact.envs import dm_control_wrapper

def compute_speed(task, physics):
        xvel = task._walker.observables.torso_xvel(physics)
        yvel = task._walker.observables.torso_yvel(physics)
        speed = np.linalg.norm([xvel, yvel])
       
        return speed

#previous MocapContinuationGym    
class MocapSpeedGym(dm_control_wrapper.DmControlWrapper):
    """
    Wraps the MultiClipMocapTracking into a Gym env.
    """

    def __init__(
        self,
        dataset: Optional[types.ClipCollection] = None,
        ref_steps: Tuple[int] = (0,),
        mocap_path: Optional[Union[str, Path]] = None,
        task_kwargs: Optional[Dict[str, Any]] = None,
        environment_kwargs: Optional[Dict[str, Any]] = None,
        act_noise: float = 0.,
        enable_all_proprios: bool = False,
        enable_cameras: bool = False,
        include_clip_id: bool = False,
        target_speed: float = 4,
        speed_error_threshold = 1,
        # for rendering
        width: int = 640,
        height: int = 480,
        camera_id: int = 3
    ):
        self._dataset = dataset or types.ClipCollection(ids=['CMU_016_22'])
        self._enable_all_proprios = enable_all_proprios
        self._enable_cameras = enable_cameras
        self._include_clip_id = include_clip_id
        self._target_speed = target_speed
        self._speed_error_threshold = speed_error_threshold
        task_kwargs = task_kwargs or dict()
        task_kwargs['ref_path'] = mocap_path if mocap_path else cmu_mocap_data.get_path_for_cmu(version='2020')
        task_kwargs['dataset'] = self._dataset
        task_kwargs['ref_steps'] = ref_steps
        super().__init__(
            tracking_continuation.MocapClipContinuation,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            act_noise=act_noise,
            width=width,
            height=height,
            camera_id=camera_id
        )
        
    def set_target_speed(self, target_speed):
        self._target_speed = target_speed

    def _get_walker(self):
        return cmu_humanoid.CMUHumanoidPositionControlledV2020

    def _create_env(
        self,
        task_type,
        task_kwargs,
        environment_kwargs,
        act_noise=0.,
        arena_size=(8., 8.)
    ):
        env = super()._create_env(task_type, task_kwargs, environment_kwargs, act_noise, arena_size)
        walker = env._task._walker
        if self._enable_all_proprios:
            walker.observables.enable_all()
            walker.observables.prev_action.enabled = False # this observable is not implemented
            if not self._enable_cameras:
                # TODO: procedurally find the cameras
                walker.observables.egocentric_camera.enabled = False
                walker.observables.body_camera.enabled = False
            env.reset()
        return env

    def _create_observation_space(self) -> spaces.Dict:
        obs_spaces = dict()
        for k, v in self._env.observation_spec().items():
            if v.dtype == np.float64 and np.prod(v.shape) > 0:
                obs_spaces[k] = spaces.Box(
                    -np.infty,
                    np.infty,
                    shape=(np.prod(v.shape),),
                    dtype=np.float32
                )
            elif v.dtype == np.uint8:
                tmp = v.generate_value()
                obs_spaces[k] = spaces.Box(
                    v.minimum.item(),
                    v.maximum.item(),
                    shape=tmp.shape,
                    dtype=np.uint8
                )
            elif k == 'walker/clip_id' and self._include_clip_id:
                obs_spaces[k] = spaces.Discrete(len(self._dataset.ids))
        return spaces.Dict(obs_spaces)

    def check_for_speed(self, time_step):
        speed = compute_speed(self._env.task, self._env.physics)
        if self._target_speed < 0:
            if speed < self._speed_error_threshold:
                obs, time_step = self.reset()
                return speed, True, time_step  
        elif abs(speed - self._target_speed) < self._speed_error_threshold:
                self._env._reset_next_step = True
                obs, time_step = self.reset()
                return speed, True, time_step
        
        self._env._reset_next_step = False
        return speed, False, time_step

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:                   
        obs, reward, done, info = super().step(action)
        
               
        if "walker/time_in_clip" in obs:
            obs["walker/time_in_clip"] = np.array([0.3])
        
        info['time_in_clip'] = obs['walker/time_in_clip'].item()
        info['start_time_in_clip'] = self._start_time_in_clip
        info['last_time_in_clip'] = self._last_time_in_clip
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        
        obs = self.get_observation(time_step)
        self._start_time_in_clip = obs['walker/time_in_clip'].item()
        self._last_time_in_clip = self._env.task._last_step / (len(self._env.task._clip_reference_features['joints'])-1)
        return obs, time_step