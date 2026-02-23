from typing import Dict, Tuple

import carla
import gym
import numpy as np
from gym import spaces

import time
from functools import wraps
from typing import Callable, Dict, List, Union, Tuple

import carla
import numpy as np

import math

from .carla_base_env import CarlaBaseEnv

from .toolkit import Observer, WorldManager
from .carla_wpt_env import CarlaWptEnv

from abc import abstractmethod

from .toolkit import BasePlanner, TTCCalculator, get_location_distance, get_vehicle_pos, get_vehicle_velocity

from .toolkit import RandomPlanner

import random

# top of file
import pygame

LANE_START_POINTS = [[131.17, -204.71, 2.0,0,180,0]]

LANE_END_POINTS = [[238.26, 159.83, 2.0,0,-80,0]]

class CarlaTrainEnv(CarlaWptEnv):
    def __init__(self, config):

        # in CarlaLaneFollowingEnv.__init__
        self._pg_screen = None
        self._pg_font = None
        self._pg_scale = 4  # 128x128 -> 512x512

        super().__init__(config)
        self.waypoints = []
        self._slow_steps = 0  # consecutive steps below min speed

        self.lane_start_point = None          # (x, y)
        self.lane_end_point = None            # (x, y)
        self._lane_pair_index = None          # selected index
        self._lane_start_transform = None     # carla.Transform for spawn
        self.goal = None                      # (x, y) terminal target

        self.vehicle_densities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self._density_index = 0


        self.max_distance_from_center = config.get("max_distance", 5.0)

        # Speed configuration: prefer meters/second in config (min_speed_mps).
        # For backward compatibility support legacy min_speed_kmh if present.
        if "min_speed_mps" in config:
            self.min_speed_mps = float(config.get("min_speed_mps", 0.5))
        else:
            # legacy key in km/h -> convert to m/s
            self.min_speed_mps = float(config.get("min_speed_kmh", 0.5)) / 3.6

        # max speed: config may specify km/h; keep both representations
        self.max_speed_kmh = float(config.get("max_speed_kmh", 100.0))
        self.max_speed_mps = self.max_speed_kmh / 3.6

        self.min_speed_timeout = float(config.get("min_speed_timeout", 10.0))

        self._offroad_steps = 0

        # derive fps from world.fixed_delta_seconds when available, otherwise fallback
        try:
            world_cfg = self._config.get("world", {}) if isinstance(self._config, dict) else getattr(self._config, "world", {})
            fixed_delta = world_cfg.get("fixed_delta_seconds", None)
            if fixed_delta:
                self._fps = float(1.0 / float(fixed_delta))
            else:
                self._fps = 14.0
        except Exception:
            self._fps = 14.0

        self.offroad_patience_steps = max(1, int(self._config.get("offroad_patience_s", 0.2) * self._fps))

        self._env_step = 0

        self._change_town_step = 20000
        self._town_changed = False

        self.v_p_count = 0      #  ❱❱  new
        self.v_count = 0        #  ❱❱  new

        # spectator view config
        self.spectator_enable = self._config.get("spectator_enable", True)
        self.spectator_mode = self._config.get("spectator_mode", "chase")   # "chase" or "topdown"
        self.spectator_distance = self._config.get("spectator_distance", 8.0)
        self.spectator_height = self._config.get("spectator_height", 3.0)
        self.spectator_pitch = self._config.get("spectator_pitch", -10.0)

        self._shield_ttl = 0

        self.max_episode_steps = int(
            config.get("max_episode_steps", int(config.get("time_limit_s", 120.0) * self._fps))
        )

        self._end_pair_index = None

        self.goal_radius = float(config.get("goal_radius", 10.0))
        self.allowed_end_indices: List[int] = []


    def _ensure_pygame(self, h, w):
        if self._pg_screen is None:
            pygame.init()
            self._pg_screen = pygame.display.set_mode((w*self._pg_scale, h*self._pg_scale))
            pygame.display.set_caption("CarlaLaneFollowingEnv")
            self._pg_font = pygame.font.SysFont("monospace", 18)

    def _extract_frame(self, obs):
        # pick the first available image-like obs
        for k in ("camera", "semantic_segmentation", "birdeye_wpt", "birdeye_with_traffic_lights"):
            if k in obs:
                frame = obs[k]
                break
        else:
            # nothing to draw
            return None
        # handle possible batch dim
        if frame.ndim == 4:
            frame = frame[0]
        # ensure uint8 HxWx3
        if frame.dtype != np.uint8:
            frame = np.clip(frame * (255.0 if frame.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return frame

    # def _blit_text(self, surface, lines, x=8, y=8, line_h=20):
    #     # translucent panel
    #     pad = 6
    #     maxw = max(self._pg_font.size(s)[0] for s in lines) if lines else 0
    #     rect = pygame.Surface((maxw + 2*pad, len(lines)*line_h + 2*pad), pygame.SRCALPHA)
    #     rect.fill((0, 0, 0, 160))
    #     surface.blit(rect, (x- pad, y- pad))
    #     # text
    #     for i, s in enumerate(lines):
    #         img = self._pg_font.render(s, True, (255, 255, 255))
    #         surface.blit(img, (x, y + i*line_h))

    def _blit_text(self, surface, lines, x=8, y=8, line_h=20, pad=6, return_size=False):
        maxw = max(self._pg_font.size(s)[0] for s in lines) if lines else 0
        w = maxw + 2*pad
        h = len(lines)*line_h + 2*pad
        rect = pygame.Surface((w, h), pygame.SRCALPHA)
        rect.fill((0, 0, 0, 160))
        surface.blit(rect, (x - pad, y - pad))
        for i, s in enumerate(lines):
            img = self._pg_font.render(s, True, (255, 255, 255))
            surface.blit(img, (x, y + i*line_h))
        return (w, h) if return_size else None


    def _render_pygame(self, obs, info):
        frame = self._extract_frame(obs)
        if frame is None:
            return
        h, w = frame.shape[:2]
        self._ensure_pygame(h, w)

        # Pygame expects WxH, so transpose
        surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        surf = pygame.transform.smoothscale(surf, (w*self._pg_scale, h*self._pg_scale))
        self._pg_screen.blit(surf, (0, 0))

        # overlay lines (add more as needed)
        action_idx = None

        # Support multiple action formats passed in `info['action']`:
        # - scalar integer (raw discrete index)
        # - 1-element array containing an integer
        # - one-hot / probability vector (argmax to get index)
        if "action" in info:
            a_raw = info["action"]
            try:
                a_arr = np.asarray(a_raw)
                flat = a_arr.reshape(-1)
                if flat.size == 1:
                    # Single value: only treat as an index if it's integer-like.
                    v = flat[0]
                    if np.issubdtype(flat.dtype, np.integer):
                        action_idx = int(v)
                    else:
                        # If it's a float that is effectively an integer (e.g. 2.0), accept it.
                        try:
                            vf = float(v)
                            if abs(vf - round(vf)) < 1e-6 and vf >= 0:
                                action_idx = int(round(vf))
                            else:
                                action_idx = None
                        except Exception:
                            action_idx = None
                else:
                    # Multi-element array: assume one-hot or prob vector and take argmax.
                    action_idx = int(np.argmax(flat))
            except Exception:
                action_idx = None

        lane_idx = info.get("lane_pair_index", getattr(self, "_lane_pair_index", None))
        end_idxs = info.get("end_pair_indices", getattr(self, "allowed_end_indices", []))


        lines = [
            f"t={self._time_step}",
            f"env_step={self._env_step}",
            f"lane_start_idx={lane_idx if lane_idx is not None else 'NA'}",
            f"end_idxs={end_idxs if end_idxs else 'NA'}",  # <--- show both accepted ends
            f"speed_kmh={info.get('speed_kmh', 0):.1f}",
            f"collision={bool(info.get('is_collision', False))}",
            f"off_road={bool(info.get('is_off_road', False))}",
            f"too_slow={bool(info.get('too_slow', False))}",
            f"dest_reached={bool(info.get('is_destination_reached', False))}",
            f"action_idx={action_idx if action_idx is not None else 'NA'}",
        ]
        # Add whether the ego is following the next waypoint (close and roughly aligned)
        try:
            wpt_dist = float(info.get('wpt_dis', 0.0))
            speed_parallel = float(info.get('speed_parallel', 0.0))
            # Consider 'following' if within 3 meters and forward speed along wpt > 0.5 m/s
            following_wpt = (wpt_dist <= 3.0) and (speed_parallel > 0.5)
        except Exception:
            following_wpt = False
        lines.append(f"following_wpt={int(following_wpt)} wpt_dist={wpt_dist:.2f}")

        # Show reward component breakdown if available in info
        # Show reward component breakdown on separate lines if available
        try:
            r_waypoints = float(info.get('r_waypoints', 0.0))
            r_speed = float(info.get('r_speed', 0.0))
            r_collision = float(info.get('r_collision', 0.0))
            r_out_of_lane = float(info.get('r_out_of_lane', 0.0))
            time_penalty = float(info.get('time_penalty', 0.0))
            total_reward = float(info.get('total_reward', 0.0))
            lines.append(f"rew_total={total_reward:.2f}")
            lines.append(f"r_waypoint={r_waypoints:.2f}")
            lines.append(f"r_speed={r_speed:.2f}")
            lines.append(f"r_collision={r_collision:.2f}")
            lines.append(f"r_out_of_lane={r_out_of_lane:.2f}")
            lines.append(f"time_penalty={time_penalty:.2f}")
        except Exception:
            # silently skip if reward keys missing
            pass
        # self._blit_text(self._pg_screen, lines)

        # keep window responsive
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        from dreamerv3 import shield_bus as sb
        s = sb.get()
        # self._blit_text(
        #     self._pg_screen,
        #     [
        #         f"SHIELD: λ={getattr(s,'lam',0):.3f}  unsafe={int(s.unsafe)}",
        #         f"{getattr(s,'orig_idx',-1)}→{getattr(s,'idx',-1)}",
        #         f"risk {getattr(s,'orig_risk',0):.3f}→{getattr(s,'chosen_risk',0):.3f}",
        #     ],
        #     x=8, y=8 + 9*20
        # )

        # first panel
        _, h1 = self._blit_text(self._pg_screen, lines, x=8, y=8, return_size=True)
        # second panel placed just below the first
        shield_lines = [
            f"SHIELD: λ={getattr(s,'lam',0):.3f}  unsafe={int(s.unsafe)}",
            f"{getattr(s,'orig_idx',-1)}→{getattr(s,'idx',-1)}",
            f"risk {getattr(s,'orig_risk',0):.3f}→{getattr(s,'chosen_risk',0):.3f}",
        ]
        self._blit_text(self._pg_screen, shield_lines, x=8, y=8 + h1 + 8)


        pygame.display.flip()


    @abstractmethod
    def get_ego_planner(self) -> BasePlanner:
        return self.ego_planner
    
        # --- lane pair selection and spawn helpers ---
    # def _select_lane_pair(self, index: int = None) -> None:
    #     """Pick a start/end lane pair and prepare a spawn transform."""
    #     if index is None:
    #         idx = random.randrange(len(LANE_START_POINTS))
    #     else:
    #         idx = index % len(LANE_START_POINTS)
    #     s = LANE_START_POINTS[idx]
    #     e = LANE_END_POINTS[idx]
    #     self._lane_pair_index = idx
    #     self.lane_start_point = (float(s[0]), float(s[1]))
    #     if idx < 8:
    #         self.lane_end_point = (float(e[0]), float(e[1]))
    #         self.lane_end_point2 = (float(e[0]), float(e[1]))
    #     else:
    #         self.lane_end_point = (float(e[0]), float(e[1]))
    #         self.lane_end_point2 = (float(e[0+4]), float(e[1+4]))
    #     # Build a safe spawn transform on the nearest driving waypoint
    #     start_loc = carla.Location(x=self.lane_start_point[0], y=self.lane_start_point[1], z=0.0)
    #     wp = self._world._map.get_waypoint(
    #         start_loc, project_to_road=True, lane_type=carla.LaneType.Driving
    #     )
    #     tf = wp.transform
    #     tf.location.z = max(tf.location.z, 0.5)
    #     self._lane_start_transform = tf
    #     # Set terminal goal in 2D
    #     self.goal = [self.lane_end_point, self.lane_end_point2]

    def _select_lane_pair(self, index: int = None) -> None:
        n = len(LANE_START_POINTS)
        idx = random.randrange(n) if index is None else index % n
        self._lane_pair_index = idx

        # start
        sx, sy, _ = LANE_START_POINTS[idx]
        self.lane_start_point = (float(sx), float(sy))

        # allowed ends: if start in [8..11] -> {idx, idx+4}; else {idx}
        if 8 <= idx <= 11 and (idx + 4) < len(LANE_END_POINTS):
            self.allowed_end_indices = [idx, idx + 4]
        else:
            self.allowed_end_indices = [idx]

        # build goal list (both accepted)
        goals = []
        for j in self.allowed_end_indices:
            ex, ey, _ = LANE_END_POINTS[j]
            goals.append((float(ex), float(ey)))
        self.goal = goals
        self.lane_end_point  = goals[0]
        self.lane_end_point2 = goals[1] if len(goals) > 1 else goals[0]

        # spawn transform
        start_loc = carla.Location(x=self.lane_start_point[0], y=self.lane_start_point[1], z=0.0)
        wp = self._world._map.get_waypoint(start_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        tf = wp.transform; tf.location.z = max(tf.location.z, 0.5)
        self._lane_start_transform = tf


    
    # --- spectator helpers ---
    def _update_spectator(self) -> None:
        if not getattr(self, "spectator_enable", False):
            return
        if self.ego is None:
            return
        spectator = self._world.carla_world.get_spectator()
        t = self.ego.get_transform()
        if self.spectator_mode == "topdown":
            loc = carla.Location(x=t.location.x, y=t.location.y,z=t.location.z + max(self.spectator_height, 15.0))
            rot = carla.Rotation(pitch=-90.0, yaw=t.rotation.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(loc, rot))
        else:  # "chase"
            fwd = t.get_forward_vector()
            loc = carla.Location(
                x=t.location.x - fwd.x * self.spectator_distance,
                y=t.location.y - fwd.y * self.spectator_distance,
                z=t.location.z + self.spectator_height
            )
            rot = carla.Rotation(pitch=self.spectator_pitch, yaw=t.rotation.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(loc, rot))  

    def on_step(self) -> None:
        self.waypoints, self.planner_stats = self.get_ego_planner().run_step()
        self.num_completed = self.planner_stats["num_completed"]

    def _set_next_vehicle_density(self):
        for actor in self._world.carla_world.get_actors():
            if actor.id != self.ego.id:
                if 'vehicle' in actor.type_id:
                    actor.destroy()
        self._density_index = (self._density_index + 1) % len(self.vehicle_densities)
        new_density = self.vehicle_densities[self._density_index]
        print(f"[Vehicle Density] Spawning {new_density} traffic vehicles.")
        self._world.spawn_auto_actors(new_density)

    def _set_random_vehicle_density(self):
        for actor in self._world.carla_world.get_actors():
            if actor.id != self.ego.id and 'vehicle' in actor.type_id:
                actor.destroy()

        new_density = random.choice(self.vehicle_densities)

        print(f"[Vehicle Density] Spawning {new_density} traffic vehicles.")
        self._world.spawn_auto_actors(new_density)

    def is_off_road(self) -> bool:
        ego_loc = self.ego.get_location()

        wp_any = self._world._map.get_waypoint(ego_loc, project_to_road=False)
        if wp_any is None:
            return True
        if not (wp_any.lane_type & carla.LaneType.Driving):
            return True

        if self.max_distance_from_center is not None:
            wp_drv = self._world._map.get_waypoint(
                ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if wp_drv is None:
                return True
            # right-vector with fallback
            if hasattr(wp_drv.transform, "get_right_vector"):
                r = wp_drv.transform.get_right_vector()
                rx, ry = r.x, r.y
            else:
                f = wp_drv.transform.get_forward_vector()
                rx, ry = f.y, -f.x  # right = forward × up
            dx = ego_loc.x - wp_drv.transform.location.x
            dy = ego_loc.y - wp_drv.transform.location.y
            cte = abs(dx * rx + dy * ry)
            if cte > self.max_distance_from_center:
                return True
        return False

    def on_reset(self):
        self.num_completed = 0
        self._time_step = 0
        self.r_speed = 0.0
        self.r_velocity = 0
        self.throttle = 0
        self.brake = 0

        self.off_road = False

        self.off_road_counter = 0

        self.is_no_velocity = False

        self._slow_steps = 0
        self._offroad_steps = 0
        self.off_road = False
        self.off_road_counter = 0

        self.ego = self._world.spawn_actor()

        self._select_lane_pair(index=self._config.get("lane_pair_index", None))

        if self._lane_start_transform is not None:
            self.ego.set_transform(self._lane_start_transform)
        # place spectator initially
        self._update_spectator()    

        self._world.spawn_auto_actors(self._config.num_vehicles)

        self._update_waypoints()

        self.r_speed = 0.0
        self.r_velocity = 0
        self.throttle = 0
        self.brake = 0
        self.off_road = False
        self.off_road_counter = 0
        self.is_no_velocity = False

        self.ego_planner = RandomPlanner(vehicle=self.ego)

        self.v_p_count = 0      #  ❱❱  new
        self.v_count = 0        #  ❱❱  new

        self.obs, _ = self._observer.get_observation(self.get_state())
        self._ensure_cost_keys(self._in_junction())

        # Debug print: resolved speed/time parameters
        try:
            print(f"[ENV DEBUG] min_speed_mps={getattr(self,'min_speed_mps',None)}, max_speed_mps={getattr(self,'max_speed_mps',None)}, min_speed_timeout={self.min_speed_timeout}, fps={self._fps}")
        except Exception:
            pass


    def apply_control(self, action) -> None:
        control = self.get_vehicle_control(action)
        # print(f"[CTRL] throttle={control.throttle:.2f} brake={control.brake:.2f} steer={control.steer:.2f}")
        self.get_ego_vehicle().apply_control(control)

    def get_vehicle_pos(self , vehicle: carla.Actor) -> Tuple[float, float]:
        location = vehicle.get_transform().location
        return location.x, location.y

    def get_vehicle_velocity(self, vehicle: carla.Actor) -> Tuple[float, float]:
        velocity = vehicle.get_velocity()
        return velocity.x, velocity.y
    
    def get_ego_vehicle(self) -> carla.Actor:
        return self.ego
    
    def _update_waypoints(self):
        ego_transform = self.get_ego_vehicle().get_transform()       
        ego_location = ego_transform.location              
        ego_rotation = ego_transform.rotation               

        waypoint = self._world._map.get_waypoint(
            ego_location, 
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        waypoints_list = []
        next_wp = waypoint
        num_points_ahead = 50
        step_distance = 2.0  

        for _ in range(num_points_ahead):
            next_points = next_wp.next(step_distance)
            if not next_points:
                break 
            next_wp = next_points[0]

            loc = next_wp.transform.location
            rot = next_wp.transform.rotation
            waypoints_list.append([loc.x, loc.y, rot.yaw])

        self.waypoints = waypoints_list

    # def get_state(self):
    #     return {
    #         "ego_waypoints": self.waypoints,
    #         "timesteps": self._time_step,
    #         "lane_start_point": self.lane_start_point,
    #         "lane_end_point": self.lane_end_point,
    #         "lane_pair_index": self._lane_pair_index,
    #     }
    
    def get_state(self):
        return {
            "ego_waypoints": self.waypoints,
            "timesteps": self._time_step,
            "lane_start_point": self.lane_start_point,
            "lane_end_point": self.lane_end_point,
            "lane_pair_index": self._lane_pair_index,
            "end_pair_indices": self.allowed_end_indices,   # <--
        }

    # Use the reward implementation from CarlaWptEnv (inherits reward()).
    # Keep this method only if you need to override; otherwise rely on parent.


    # def get_goal_dist(self, ego_location):
    #     if self.goal is None:
    #         return 0
    #     else:
    #         # return self.get_location_distance(ego_location, self.goal)  
    #         return any(self.get_location_distance(ego_location, g) <= 10 for g in self.goal)

    def get_goal_dist(self, ego_location):
        if not self.goal:
            return float("inf")
        return min(self.get_location_distance(ego_location, g) for g in self.goal)
        

    def get_location_distance(self, loc1, loc2):
        dx = loc1[0] - loc2[0]
        dy = loc1[1] - loc2[1]
        return math.sqrt(dx*dx + dy*dy)

    def _in_junction(self):
        wp = self._world._map.get_waypoint(
            self.ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
            )
        return bool(wp and wp.is_junction)

    def is_wrong_direction(self) -> bool:
        wp = self._world._map.get_waypoint(
            self.ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if wp is None or wp.is_junction:
            return False
        f_lane = wp.transform.get_forward_vector()
        f_car  = self.ego.get_transform().get_forward_vector()
        dot = f_lane.x*f_car.x + f_lane.y*f_car.y + f_lane.z*f_car.z
        margin = float(self._config.get("wrong_dir_margin", 0.05))
        return dot < -margin
    
    def get_terminal_conditions(self) -> Dict[str, bool]:
        ego_location = np.array(self.get_vehicle_pos(self.get_ego_vehicle()))

        if self.v_count == 10:
            self.is_no_velocity = True
        else:
            self.is_no_velocity = False

        if self.off_road_counter == 1:
            self.off_road = True

        time_exceeded = (self._time_step >= self.max_episode_steps)
        # Terminate if speed stayed below threshold for the configured duration
        threshold_steps = int(self.min_speed_timeout * self._fps)
        too_slow = self._slow_steps >= threshold_steps
        offroad_terminal = self._offroad_steps >= self.offroad_patience_steps

        return {
            "is_collision": self.is_collision(),
            "is_off_road": offroad_terminal,
            "is_wrong_direction": self.is_wrong_direction(),
            "time_exceeded": time_exceeded,
            "too_slow": too_slow,
            "is_destination_reached": self.is_destination_reached(),
        }

    def is_collision(self) -> bool:
        """Check if the ego vehicle is in collision."""
        return self.obs["collision"][0] > 0
    
    # def is_destination_reached(self):
    #     return self.goal is not None and \
    #         self.get_goal_dist(self.get_vehicle_pos(self.get_ego_vehicle())) <= 10
    #     # return self.get_goal_dist(self.get_vehicle_pos(self.get_ego_vehicle())) <= 10

    # def is_destination_reached(self):
    #     if not self.goal:
    #         return False
    #     ego_xy = self.get_vehicle_pos(self.get_ego_vehicle())
    #     return any(self.get_location_distance(ego_xy, g) <= 10 for g in self.goal)

    def is_destination_reached(self):
        if not self.goal:
            return False
        ego_xy = self.get_vehicle_pos(self.get_ego_vehicle())
        return any(self.get_location_distance(ego_xy, g) <= self.goal_radius for g in self.goal)

    def _ensure_cost_keys(self, in_junc: bool):
        # lane_invasion -> (1,) float32
        li = float(np.squeeze(self.obs.get("lane_invasion", 0.0)))
        li = np.array([li], np.float32)
        self.obs["lane_invasion"]     = li
        # self.obs["lane_invasion_raw"] = li.copy()
        # offlane is the gated version used by the model/shield
        # self.obs["offlane"] = np.array([0.0 if in_junc else float(li[0])], np.float32)

    def step(self, action):
        self.apply_control(action)
        self._world.step()
        self._time_step += 1
        self._env_step += 1
        
        # update spectator every tick
        self._update_spectator()

        if (self._time_step % 1000) == 0 and self._time_step > 0:

            for actor in self._world.carla_world.get_actors():
                if actor.id != self.ego.id and 'vehicle' in actor.type_id:
                    actor.destroy()

            self._density_index = (self._density_index + 1) % len(self.vehicle_densities)
            new_density = self.vehicle_densities[self._density_index]
            self._world.spawn_auto_actors(new_density)
            print(f"[Vehicle Density] Spawning {new_density} traffic vehicles.")

        self._update_waypoints()

        env_state = self.get_state()

        vx, vy = self.get_vehicle_velocity(self.get_ego_vehicle())
        speed_ms = math.hypot(vx, vy)
        speed_kmh = speed_ms * 3.6

        # Track consecutive slow steps using min_speed_mps (config may supply m/s)
        if speed_ms < getattr(self, "min_speed_mps", 0.5):
            self._slow_steps += 1
        else:
            self._slow_steps = 0

        # after speed / _slow_steps update
        if self.is_off_road():
            self._offroad_steps += 1
        else:
            self._offroad_steps = 0


        if speed_ms > 0.5:
            self.v_p_count += 1

        if self.v_p_count >= 100:
            self.v_count = 0


        # if np.linalg.norm(ego_velocity) <= 0.2:
        if speed_ms <= 0.5:
            self.v_count += 1
        else:
            self.v_count = 0

        
        self.obs, obs_info = self._observer.get_observation(env_state)

        in_junc = self._in_junction()
        self._ensure_cost_keys(in_junc)


        if in_junc:
            self._slow_steps = 0
    
        is_terminal, terminal_conds = self._is_terminal()
        reward, reward_info = self.reward()

        info = {
            **env_state,
            **terminal_conds,
            **obs_info,
            **reward_info,
            "action": action,
            "time_steps": self._time_step,
            "env_step": self._env_step,
        }
        if self._config.eval:
            info = {f"eval_{k}": v for k, v in info.items()}
            self.obs = {**self.obs, **info}
        info["speed_kmh"] = speed_kmh  # so HUD can show it

        if self._config.display.enable:
            self._render_pygame(self.obs, info)

        info["end_pair_indices"] = self.allowed_end_indices

        # --- per-step CSV write ---
        try:
            self._log_csv_step(reward, is_terminal, info | reward_info)
        except Exception:
            pass


        return (self.obs, reward, is_terminal, info)
