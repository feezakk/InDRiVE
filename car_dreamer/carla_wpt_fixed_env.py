from collections import deque

import carla
import numpy as np

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedPathPlanner, get_location_distance, get_vehicle_pos


class CarlaWptFixedEnv(CarlaWptEnv):
    """
    This is the base env for all waypoint following tasks with a fixed route and car flow.
    **DO NOT** instantiate this class directly.

    All envs that inherit from this class also inherits the following config parameters:

    * ``lane_start_point``: The starting point of the ego vehicle in ``[x, y, z, yaw]``
    * ``ego_path``: The fixed path for the ego vehicle in array of ``[x, y, z]``
    * ``use_road_waypoints``: For each segment, whether to adapt the path according to road or use straight line
    * ``flow_spawn_point``: The spawn point of the car flow in ``[x, y, z, yaw]``
    * ``min_flow_dist``: Minimum distance between two cars in the flow, if ``None``, no cars will be spawned
    * ``max_flow_dist``: Maximum distance between two cars in the flow

    """

    # def on_reset(self) -> None:
    #     self.ego_src = self._config.lane_start_point
    #     ego_transform = carla.Transform(carla.Location(*self.ego_src[:3]), carla.Rotation(yaw=self.ego_src[3]))
    #     self.ego = self._world.spawn_actor(transform=ego_transform)
    #     sp = self._config.lane_start_point
    #     ep = self._config.ego_path
    #     # self.ego_path = self._config.ego_path
    #     if isinstance(sp[0], (list, tuple)) and isinstance(ep[0], (list, tuple)) and isinstance(ep[0][0], (list, tuple)):
    #         idx = np.random.randint(len(sp))
    #         self.ego_src = sp[idx]                     # must be [x,y,z,yaw]
    #         self.ego_path = ep[idx]                    # list of [x,y,z]
    #     else:
    #         self.ego_src = sp
    #         self.ego_path = ep

    #     self.use_road_waypoints = self._config.use_road_waypoints
    #     self.ego_planner = FixedPathPlanner(
    #         vehicle=self.ego,
    #         vehicle_path=self.ego_path,
    #         use_road_waypoints=self.use_road_waypoints,
    #     )
    #     self.waypoints, self.planner_stats = self.ego_planner.run_step()
    #     self.num_completed = self.planner_stats["num_completed"]

    #     # # Initialize car flow
    #     # self.actor_flow = deque()
    #     # flow_spawn_point = self._config.flow_spawn_point
    #     # self.flow_transform = carla.Transform(
    #     #     carla.Location(*flow_spawn_point[:3]),
    #     #     carla.Rotation(yaw=flow_spawn_point[3]),
    #     # )

    def on_reset(self) -> None:
        # 1) Pick one route if lists are provided
        sp = self._config.lane_start_point
        ep = self._config.ego_path
        if isinstance(sp[0], (list, tuple)):           # list of starts
            idx = np.random.randint(len(sp))
            sp = sp[idx]
            ep = ep[idx]
        # now sp is one start pose, ep is one path (list of [x,y,z])

        # 2) Ensure yaw; if missing, take it from the road waypoint
        if len(sp) == 3:
            loc = carla.Location(float(sp[0]), float(sp[1]), float(sp[2]))
            wp = self._world._map.get_waypoint(loc, project_to_road=True,
                                            lane_type=carla.LaneType.Driving)
            yaw = float(wp.transform.rotation.yaw) if wp else 0.0
            sp = [float(sp[0]), float(sp[1]), float(sp[2]), yaw]
        else:
            sp = [float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3])]

        if isinstance(self._config.lane_start_point[0], (list, tuple)):
            self._lane_pair_index = int(idx)
            # if each start has exactly one end, just echo the same index
            self.allowed_end_indices = [int(idx)]
        else:
            self._lane_pair_index = 0
            self.allowed_end_indices = [0]

        # 3) Spawn ego
        ego_tf = carla.Transform(carla.Location(*sp[:3]), carla.Rotation(yaw=sp[3]))
        self.ego = self._world.spawn_actor(transform=ego_tf)

        self._update_spectator()


        # 4) Planner setup
        self.ego_path = [[float(p[0]), float(p[1]), float(p[2])] for p in ep]
        self.use_road_waypoints = self._config.use_road_waypoints
        self.ego_planner = FixedPathPlanner(
            vehicle=self.ego,
            vehicle_path=self.ego_path,
            use_road_waypoints=self.use_road_waypoints,
        )
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]

        


    def on_step(self) -> None:
        super().on_step()

        # # Generate and sink car flow
        # spawn = False
        # if "min_flow_dist" in self._config:
        #     if len(self.actor_flow) == 0:
        #         spawn = True
        #     else:
        #         spawn_location = np.array(self._config.flow_spawn_point[:2])
        #         nearest_car_location = np.array(get_vehicle_pos(self.actor_flow[-1]))
        #         flow_dist = np.random.uniform(self._config.min_flow_dist, self._config.max_flow_dist)
        #         if get_location_distance(spawn_location, nearest_car_location) >= flow_dist:
        #             spawn = True
        # if spawn:
        #     vehicle = self._world.try_spawn_aggresive_actor(self.flow_transform)
        #     if vehicle is not None:
        #         self.actor_flow.append(vehicle)
