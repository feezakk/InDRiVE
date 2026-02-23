from abc import abstractmethod
from typing import Dict, Tuple

import numpy as np
import carla

from .carla_base_env import CarlaBaseEnv
from .toolkit import BasePlanner, TTCCalculator, get_location_distance, get_vehicle_pos, get_vehicle_velocity

import csv, os, time


class CarlaWptEnv(CarlaBaseEnv):
    """Waypoint‑following base. Do not instantiate directly."""

    @abstractmethod
    def get_ego_planner(self) -> BasePlanner:
        return self.ego_planner

    def get_state(self):
        return {
            "ego_waypoints": getattr(self, "waypoints", []),
            "timesteps": self._time_step,
            "lane_pair_index": getattr(self, "_lane_pair_index", None),
            "end_pair_indices": getattr(self, "allowed_end_indices", []),
        }

    def apply_control(self, action) -> None:
        self.get_ego_vehicle().apply_control(self.get_vehicle_control(action))

    def on_step(self) -> None:
        self.waypoints, self.planner_stats = self.get_ego_planner().run_step()
        self.num_completed = self.planner_stats["num_completed"]

    def _lane_center_error(self, loc=None) -> Tuple[float, float]:
        if loc is None:
            loc = self.get_ego_vehicle().get_location()
        ego_wp = self._world.carla_map.get_waypoint(
            loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if ego_wp is None:
            return float("inf"), 0.0  # treat as off-road
        yaw = np.deg2rad(ego_wp.transform.rotation.yaw)
        R = np.array([[ np.cos(yaw),  np.sin(yaw)],
                    [-np.sin(yaw),  np.cos(yaw)]], dtype=np.float32)  # rotate global->lane
        wp_xy  = np.array([ego_wp.transform.location.x, ego_wp.transform.location.y], dtype=np.float32)
        ego_xy = np.array([loc.x, loc.y], dtype=np.float32)
        lat_err = float(abs((R @ (ego_xy - wp_xy))[1]))
        lane_w = float(max(ego_wp.lane_width, 0.1))
        return lat_err, lane_w

    # ------- Reward with pretrain gating -------
    def reward(self):
        cfg = self._config.reward
        reward_scales = cfg.scales
        ego = self.get_ego_vehicle()
        ego_location = np.array([*get_vehicle_pos(ego)])
        ego_velocity = np.array([*get_vehicle_velocity(ego)])
        speed_norm = float(np.linalg.norm(ego_velocity))

        # Reward for reaching waypoints
        r_waypoints = 0.0
        if self.num_completed > 0:
            r_waypoints = reward_scales["waypoint"]

        # Reward for speed
        r_speed = 0.0
        speed_parallel = 0.0
        speed_perpendicular = 0.0
        if len(self.waypoints) > 0:
            # compute the wpt line direction
            next_waypoint = self.waypoints[0]
            next_location = np.array([next_waypoint[0], next_waypoint[1]])
            yaw_radius = next_waypoint[2] * np.pi / 180
            waypoint_direction = np.array([np.cos(yaw_radius), np.sin(yaw_radius)])

            # compute the perpendicular direction
            goal_offset = next_location - ego_location
            perp_direction = goal_offset - np.dot(goal_offset, waypoint_direction) * waypoint_direction
            perp_direction_norm = np.linalg.norm(perp_direction)
            if perp_direction_norm > 0.05:
                perp_direction = perp_direction / perp_direction_norm
            else:
                perp_direction = np.array([0.0, 0.0])

            # compute the speed reward
            desired_speed = self._config.reward.desired_speed
            speed_parallel = np.dot(ego_velocity, waypoint_direction)
            speed_perpendicular = np.abs(np.dot(ego_velocity, perp_direction))
            r_speed = (desired_speed - np.abs(speed_parallel - desired_speed) - 2 * min(speed_perpendicular, 0.5)) * reward_scales["speed"]

        # Reward for collision
        r_collision = 0.0
        if reward_scales["collision"] > 0 and self.is_collision():
            r_collision = -reward_scales["collision"] * np.abs(speed_norm)

        # # Reward for going out of lane
        # r_out_of_lane = 0.0
        # if len(self.waypoints) > 0:
        #     dist = perp_direction_norm
        #     if dist > 0.5:
        #         r_out_of_lane = -reward_scales["out_of_lane"] * (dist - 0.5)

        # Reward for lane keeping (centerline-based, width-aware, capped)
        # r_out_of_lane = 0.0
        # if len(self.waypoints) > 0:
        #     ego_loc = self.get_ego_vehicle().get_location()
        #     ego_wp = self._world.carla_map.get_waypoint(
        #         ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        #     )
        #     if ego_wp is not None:
        #         # local frame at ego_wp: x = along lane, y = lateral
        #         yaw = np.deg2rad(ego_wp.transform.rotation.yaw)
        #         R = np.array([[ np.cos(yaw),  np.sin(yaw)],
        #                     [-np.sin(yaw),  np.cos(yaw)]], dtype=np.float32)
        #         wp_xy = np.array([ego_wp.transform.location.x, ego_wp.transform.location.y], dtype=np.float32)
        #         ego_xy = np.array([ego_loc.x, ego_loc.y], dtype=np.float32)
        #         local = R @ (ego_xy - wp_xy)
        #         lat_err = float(abs(local[1]))  # meters from lane center

        #         lane_w = float(max(ego_wp.lane_width, 0.1))
        #         tol = 0.2 * lane_w                 # no penalty inside this band
        #         margin = max(0.5 * lane_w - tol, 0.1)  # to boundary
        #         x = max(0.0, (lat_err - tol) / margin)

        #         # smooth, bounded penalty in [0, 1]
        #         penalty = min(1.0, x * x)          # quadratic; Huber or logistic also OK
        #         r_out_of_lane = -reward_scales["out_of_lane"] * penalty


        r_out_of_lane = 0.0
        if self.waypoints:  # keep gating near route end if desired
            lat_err, lane_w = self._lane_center_error()
            if np.isfinite(lat_err) and lane_w > 0.0:
                tol = 0.2 * lane_w
                margin = max(0.5 * lane_w - tol, 0.1)
                x = max(0.0, (lat_err - tol) / margin)
                penalty = min(1.0, x * x)
                r_out_of_lane = -reward_scales["out_of_lane"] * penalty


        # Reward for reaching the destination
        r_destination = 0.0
        if self.is_destination_reached():
            r_destination = reward_scales["destination_reached"]

        # Time penalty
        time_penalty = -reward_scales["time"]

        # Total reward
        total_reward = r_waypoints + r_speed + r_collision + r_out_of_lane + r_destination + time_penalty

        ttc = TTCCalculator.get_ttc(ego, self._world.carla_world, self._world.carla_map)

        info = {
            **self.planner_stats,
            "ego_x": ego_location[0],
            "ego_y": ego_location[1],
            "speed_parallel": speed_parallel,
            "speed_perpendicular": speed_perpendicular,
            "speed_norm": speed_norm,
            "wpt_dis": self.get_wpt_dist(ego_location),
            "r_waypoints": r_waypoints,
            "r_speed": r_speed,
            "r_collision": r_collision,
            "r_out_of_lane": r_out_of_lane,
            "time_penalty": time_penalty,
            "r_destination": r_destination,
            "ttc": ttc,
            "wrong_dir_steps": getattr(self, "_wrong_dir_steps", 0),
            "slow_steps": getattr(self, "_slow_steps", 0),
            "lat_err": float(lat_err) if 'lat_err' in locals() else None,
            "lane_width": float(lane_w) if 'lane_w' in locals() else None,
        }

        return total_reward, info
    
    # def reward(self):
    #     cfg = self._config.reward
    #     s = cfg.scales
    #     ego = self.get_ego_vehicle()

    #     # Velocity and speed units
    #     v = np.array([*get_vehicle_velocity(ego)], dtype=np.float32)
    #     speed = float(np.linalg.norm(v))                     # m/s
    #     desired = float(getattr(cfg, "desired_speed", 8.0))  # m/s

    #     # Next waypoint geometry
    #     if self.waypoints:
    #         wx, wy, yaw_deg = self.waypoints[0]
    #         wdir = np.array([np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))], np.float32)
    #         ex, ey = get_vehicle_pos(ego)
    #         ego_xy = np.array([ex, ey], np.float32)
    #         d_curr = float(np.linalg.norm(ego_xy - np.array([wx, wy], np.float32)))
    #         d_prev = float(getattr(self, "_prev_wpt_dist", d_curr))
    #         r_progress = s.get("progress", 0.0) * (d_prev - d_curr)   # potential-based
    #         self._prev_wpt_dist = d_curr
    #         align = float(np.dot(v / (speed + 1e-9), wdir))           # [-1,1]
    #         align = max(0.0, align)                                   # ignore backward speed
    #     else:
    #         r_progress, align = 0.0, 0.0

    #     # Speed shaping: reward near desired only when aligned; penalize overspeed
    #     # Normalize to [0,1] peak at desired
    #     sp_err = abs(speed - desired) / max(desired, 1e-3)
    #     r_speed = s.get("speed", 0.0) * (1.0 - sp_err) * align
    #     r_overspeed = -s.get("overspeed", 0.0) * max(0.0, speed - desired)

    #     # Lane keeping using center error
    #     lat_err, lane_w = self._lane_center_error()
    #     tol = 0.2 * lane_w
    #     margin = max(0.5 * lane_w - tol, 0.1)
    #     x = max(0.0, (lat_err - tol) / margin)
    #     # Huber-like: quadratic near center, linear after 1.0
    #     lane_pen = x*x if x < 1.0 else (2.0*x - 1.0)
    #     r_out_of_lane = -s.get("out_of_lane", 0.0) * lane_pen

    #     # Waypoint completions this step
    #     completed = int(self.planner_stats.get("num_completed", 0))
    #     r_waypoints = s.get("waypoint", 0.0) * completed

    #     # Collisions: constant floor + speed-scaled
    #     r_collision = 0.0
    #     if self.is_collision():
    #         r_collision = - (s.get("collision", 0.0) * (1.0 + 0.5*speed))

    #     # Optional TTC safety shaping
    #     ttc = TTCCalculator.get_ttc(ego, self._world.carla_world, self._world.carla_map)
    #     r_ttc = 0.0
    #     if np.isfinite(ttc):
    #         ttc_thr = float(getattr(cfg, "ttc_threshold_s", 2.0))
    #         if ttc < ttc_thr:
    #             r_ttc = -s.get("ttc", 0.0) * (ttc_thr - ttc) / ttc_thr

    #     # Destination bonus and time penalty
    #     r_destination = s.get("destination_reached", 0.0) if self.is_destination_reached() else 0.0
    #     time_penalty = -s.get("time", 0.0)

    #     total_reward = r_progress + r_speed + r_overspeed + r_out_of_lane + r_waypoints + r_collision + r_ttc + r_destination + time_penalty

    #     info = {
    #         **self.planner_stats,
    #         "speed_norm": speed,
    #         "wpt_dis": d_curr if self.waypoints else None,
    #         "align": align,
    #         "r_progress": r_progress,
    #         "r_speed": r_speed,
    #         "r_overspeed": r_overspeed,
    #         "r_collision": r_collision,
    #         "r_out_of_lane": r_out_of_lane,
    #         "r_destination": r_destination,
    #         "time_penalty": time_penalty,
    #         "r_ttc": r_ttc,
    #         "ttc": ttc,
    #         "lat_err": float(lat_err),
    #         "lane_width": float(lane_w),
    #         "total_reward": float(total_reward),
    #     }
    #     return total_reward, info



    
    def _is_off_road(self, ego) -> bool:
        loc = ego.get_location()
        try:
            wp = self._world.carla_map.get_waypoint(
                loc, project_to_road=False, lane_type=carla.LaneType.Driving
            )
            if wp is None:
                return True
            return False
        except Exception:
            ego_xy = (loc.x, loc.y)
            return self.get_wpt_dist(ego_xy) > getattr(self._config.terminal, "off_road_dist_thres", 4.0)
       

    def is_destination_reached(self):
        # Fallback: near end of waypoint queue
        # return len(getattr(self, "waypoints", [])) <= 5

        # print("********" , self.goal)

        gx, gy = float(self.goal[0][0]), float(self.goal[0][1])

        ex, ey = get_vehicle_pos(self.get_ego_vehicle())
        dist = float(np.hypot(ex - gx, ey - gy))
        tol = float(getattr(self._config.terminal, "dest_tolerance_m", 2.0))
        return dist <= tol

    def get_terminal_conditions(self):
        term = self._config.terminal
        ego = self.get_ego_vehicle()
        ego_xy = get_vehicle_pos(self.get_ego_vehicle())

        lat_err, lane_w = self._lane_center_error()
        frac = getattr(term, "out_of_lane_fraction", 0.9)  # near boundary
        


        conds = {
            "is_collision": self.is_collision(),
            "time_exceeded": self._time_step > term.time_limit,
            # "out_of_lane": self.get_wpt_dist(ego_xy) > term.out_lane_thres,
            "out_of_lane": lat_err > 0.5 * lane_w * frac, 
            "destination_reached": self.is_destination_reached(),
        }

        # speeds
        ego_vel = np.array([*get_vehicle_velocity(ego)], dtype=np.float32)
        speed = float(np.linalg.norm(ego_vel))

        # ---- wrong direction (uses route heading + velocity direction)
        wrong_now = False
        cos_to_route = 1.0
        if getattr(self, "waypoints", []):
            yaw = float(self.waypoints[0][2]) * np.pi / 180.0
            wdir = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
            if speed > getattr(term, "wrong_dir_min_speed", 0.5):
                cos_to_route = float(np.dot(ego_vel, wdir) / (speed + 1e-9))
                ang_ok = cos_to_route >= np.cos(np.deg2rad(getattr(term, "wrong_dir_angle_deg", 120.0)))
                wrong_now = not ang_ok
        self._wrong_dir_steps = (getattr(self, "_wrong_dir_steps", 0) + 1) if wrong_now else 0
        conds["wrong_direction"] = self._wrong_dir_steps >= getattr(term, "wrong_dir_duration", 2000)

        # ---- too slow (consecutive steps below min_speed, after grace)
        slow_now = speed < getattr(term, "min_speed", 0.01) and self._time_step > getattr(term, "slow_grace_steps", 2000)
        self._slow_steps = (getattr(self, "_slow_steps", 0) + 1) if slow_now else 0
        conds["too_slow"] = self._slow_steps >= getattr(term, "too_slow_duration", 2000)

        # ---- off road (binary)
        conds["off_road"] = self._is_off_road(ego)

        # pretrain gating
        if self.stage == "pretrain":
            conds.update({
                "is_collision": False,
                "out_of_lane": False,
                "destination_reached": False,
                "wrong_direction": False,
                "too_slow": False,
                "off_road": False,
            })

        return conds

    def get_wpt_dist(self, ego_location):
        wpts = getattr(self, "waypoints", [])
        if not wpts:
            return 0.0
        x, y, _ = wpts[0]
        return get_location_distance(ego_location, (x, y))
