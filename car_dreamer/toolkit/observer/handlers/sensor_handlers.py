from abc import abstractmethod
from typing import Dict, Tuple

import carla
import numpy as np
from gym import spaces

from ...carla_manager import WorldManager
from .base_handler import BaseHandler


class SensorHandler(BaseHandler):
    """
    Base handler for sensor data endpoints.
    """

    def __init__(self, world: WorldManager, config):
        super().__init__(world, config)
        blueprint = self._world.get_blueprint(config.blueprint)
        if "transform" in config:
            self._transform = carla.Transform(carla.Location(**config.transform))
        else:
            self._transform = carla.Transform()
        if "attributes" in config:
            for attr_name, attr_value in config.attributes.items():
                blueprint.set_attribute(attr_name, str(attr_value))
        self._blueprint = blueprint
        self._sensor = None
        self._data = None

    @property
    def _default_obs_type(self) -> np.dtype:
        obs_space = self._get_observation_space()
        if isinstance(obs_space, spaces.Box):
            return obs_space.dtype
        return np.uint8

    @property
    def _default_obs(self) -> np.ndarray:
        return np.zeros(self._config.shape, dtype=self._default_obs_type)

    @abstractmethod
    def _get_observation_space(self) -> spaces.Space:
        pass

    @abstractmethod
    def _update_data(self, data) -> None:
        pass

    def get_observation_space(self) -> Dict:
        return {self._config.key: self._get_observation_space()}

    def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
        obs = {self._config.key: (self._data if self._data is not None else self._default_obs)}
        info = {}
        return obs, info

    def destroy(self) -> None:
        self._data = None
        if self._sensor is not None:
            self._sensor.destroy()

    def reset(self, ego: carla.Actor) -> None:
        self._sensor = self._world.spawn_unmanaged_actor(self._transform, self._blueprint, attach_to=ego)
        self._sensor.listen(self._update_data)


class CameraHandler(SensorHandler):
    def _get_observation_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=255, shape=self._config.shape, dtype=np.uint8)

    def _update_data(self, data) -> None:
        camera_data = np.frombuffer(data.raw_data, dtype=np.uint8)
        camera_data = np.reshape(camera_data, (data.height, data.width, 4))
        camera_data = camera_data[:, :, :3]
        camera_data = camera_data[:, :, ::-1]
        self._data = camera_data


class LidarHandler(SensorHandler):
    def __init__(self, world: WorldManager, config):
        super().__init__(world, config)
        self._obs_range = config.attributes.range
        self._lidar_z = config.transform.z
        self._lidar_bin = config.lidar_bin
        self._ego_offset = config.ego_offset

    def _update_data(self, data) -> None:
        self._data = data

    def _get_observation_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=255, shape=self._config.shape, dtype=np.uint8)

    def get_observation(self, env_state: Dict) -> Tuple[Dict]:
        if self._data is None:
            return {self._config.key: self._default_obs}, {}

        points = np.frombuffer(self._data.raw_data, dtype=np.dtype("f4")).reshape(-1, 4)
        points = points[np.linalg.norm(points[:, :3], axis=1) <= self._obs_range]
        points[1, :] = -points[1, :]

        intensities = np.interp(points[:, 3], (points[:, 3].min(), points[:, 3].max()), (0, 1))
        colors = (intensities[:, np.newaxis] * np.array([[255, 0, 0]])).astype(np.uint8)

        y_bins = np.arange(
            -(self._obs_range - self._ego_offset),
            self._ego_offset + self._lidar_bin,
            self._lidar_bin,
        )
        x_bins = np.arange(-self._obs_range / 2, self._obs_range / 2 + self._lidar_bin, self._lidar_bin)
        z_bins = [-self._lidar_z - 1, -self._lidar_z + 0.25, 1]
        lidar, _ = np.histogramdd(points[:, :3], bins=(x_bins, y_bins, z_bins))

        lidar = lidar[: self._config.shape[0], : self._config.shape[1], :2]
        ground_mask = lidar[:, :, 0] > 0
        obstacle_mask = lidar[:, :, 1] > 0

        image = np.zeros((lidar.shape[0], lidar.shape[1], 3), dtype=np.uint8)
        image[ground_mask] = colors[: ground_mask.sum()]
        image[obstacle_mask] = np.array([0, 255, 0], dtype=np.uint8)
        image = np.flip(image, axis=0)

        obs = {self._config.key: image}
        info = {}
        return obs, info


class CollisionHandler(SensorHandler):
    def __init__(self, world, config):
        super().__init__(world, config)
        self._data = np.zeros(tuple(self._config.shape), dtype=np.float32)

    def _get_observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=tuple(self._config.shape), dtype=np.float32)

    def _update_data(self, data):
        imp = data.normal_impulse
        val = float(np.sqrt(imp.x**2 + imp.y**2 + imp.z**2))
        self._data = np.full(tuple(self._config.shape), val, dtype=np.float32)
    # def _get_observation_space(self) -> spaces.Space:
    #     return spaces.Box(low=0, high=np.inf, shape=self._config.shape, dtype=np.float32)

    # def _update_data(self, data) -> None:
    #     impulse = data.normal_impulse
    #     collision_intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    #     self._data = collision_intensity * np.ones(self._config.shape)

class LaneInvasionHandler(SensorHandler):
    def __init__(self, world, config):
        super().__init__(world, config)
        self._data = np.zeros(tuple(self._config.shape), dtype=np.float32)

    def _get_observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=tuple(self._config.shape), dtype=np.float32)

    def _update_data(self, data):
        self._data = np.ones(tuple(self._config.shape), dtype=np.float32)

    # def _get_observation_space(self) -> spaces.Space:
    #     shape = tuple(self._config.shape)
    #     return spaces.Box(low=0, high=np.inf, shape=shape, dtype=np.float32)

    # def _update_data(self, data) -> None:
    #     # Update to handle lane invasion events
    #     self._data = np.ones(tuple(self._config.shape), dtype=np.float32)

# class SemanticSegmentationOneImageHandler(SensorHandler):
#     def _get_observation_space(self) -> spaces.Space:
#         return spaces.Box(low=0, high=255, shape=self._config.shape, dtype=np.uint8)

#     def _update_data(self, data) -> None:
#         camera_data = np.frombuffer(data.raw_data, dtype=np.uint8)
#         camera_data = np.reshape(camera_data, (data.height, data.width, 4))
#         camera_data = camera_data[:, :, :3]
#         camera_data = camera_data[:, :, ::-1]
#         self._data = camera_data

class SemanticSegmentationHandler(SensorHandler):
    def _get_observation_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=255, shape=self._config.shape, dtype=np.uint8)

    def _update_data(self, data) -> None:
        data.convert(carla.ColorConverter.CityScapesPalette)
        camera_data = np.frombuffer(data.raw_data, dtype=np.uint8)
        camera_data = np.reshape(camera_data, (data.height, data.width, 4))
        camera_data = camera_data[:, :, :3]
        camera_data = camera_data[:, :, ::-1]
        self._data = camera_data

    # def __init__(self, world, config):
    #     super().__init__(world, config)
    #     from collections import deque
    #     self._data_buffer = deque(maxlen=4)
    #     H, W, C = self._config.shape  # C should be 12
    #     self._data = np.zeros((H, W, C), np.uint8)

    # def _get_observation_space(self):
    #     return spaces.Box(low=0, high=255, shape=tuple(self._config.shape), dtype=np.uint8)

    # def _update_data(self, data):
    #     data.convert(carla.ColorConverter.CityScapesPalette)
    #     frm = np.frombuffer(data.raw_data, np.uint8).reshape(data.height, data.width, 4)[:, :, :3][:, :, ::-1]
    #     self._data_buffer.append(frm)
    #     H, W, _ = frm.shape
    #     frames = list(self._data_buffer)
    #     while len(frames) < 4:
    #         frames.insert(0, np.zeros((H, W, 3), np.uint8))
    #     self._data = np.concatenate(frames[-4:], axis=-1)  # (H,W,12)

    # def get_observation(self, _):
    #     return {self._config.key: self._data}, {}

    # def __init__(self, world: WorldManager, config):
    #     super().__init__(world, config)

    #     from collections import deque  # Import deque for frame buffer
    #     self._data_buffer = deque(maxlen=4)  # Buffer to store the last 4 frames
    #     self._data = np.zeros(tuple(self._config.shape), dtype=np.uint8)  # (H,W,12)

    # def _get_observation_space(self) -> spaces.Space:
    #     # Expect stacked frames along channel axis as configured (e.g., [H, W, 12])
    #     return spaces.Box(low=0, high=255, shape=tuple(self._config.shape), dtype=np.uint8)

    # def _update_data(self, data) -> None:
    #     data.convert(carla.ColorConverter.CityScapesPalette)
    #     # frame = np.frombuffer(data.raw_data, dtype=np.uint8)
    #     # frame = np.reshape(frame, (data.height, data.width, 4))[:, :, :3][:, :, ::-1]
    #     arr = np.frombuffer(data.raw_data, dtype=np.uint8).reshape(data.height, data.width, 4)[:, :, :3][:, :, ::-1]
    #     self._data_buffer.append(arr)

    #     frames = list(self._data_buffer)
    #     while len(frames) < 4:
    #         frames.insert(0, np.zeros_like(arr))
    #     self._data = np.concatenate(frames[-4:], axis=-1)  # (H,W,12)
    #     # # Keep raw 3-channel frames only in the buffer; do not expose 3-channel frames directly.
    #     # self._data = None
    #     # self._data_buffer.append(frame)

    # def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
    #     # Always return a fixed shape per config, padding with zeros if needed.
    #     # h, w, c = tuple(self._config.shape)
    #     # assert c % 3 == 0, "semantic_segmentation shape must be multiple of 3 in channels"
    #     # needed = c // 3
    #     # frames = list(self._data_buffer)[-needed:]
    #     # if len(frames) < needed:
    #     #     pad = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(needed - len(frames))]
    #     #     frames = frames + pad
    #     # stacked_frames = np.concatenate(frames, axis=-1)

    #     obs = {self._config.key: self._data}
    #     info = {}
    #     return obs, info


    #def visualize(self):
        #if self._data is not None:
        #    render_camera(self._data)  # Display the latest camera frame

    #    if len(self._data_buffer) > 0:
            # Use the latest frame in the buffer for visualization
    #        render_camera(self._data_buffer[-1])  # Render the latest frame
