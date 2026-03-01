from typing import Tuple, Dict, List
import math, random, numpy as np, carla, pygame

from .carla_wpt_env import CarlaWptEnv
from .toolkit import BasePlanner, RandomPlanner
from .route_pool_mixin import RoutePoolMixin

from dreamerv3 import shield_bus as sb

class CarlaEvaluateEnv(RoutePoolMixin, CarlaWptEnv):
    # Default tables if YAML does not provide them

    LANE_START_POINTS = [
      [60.7, 306.7, 1.0,0,0,0], [-7.08, 209.11, 1.0,0,90,0], #Straight
      [148.41, 306.5, 1.0,0,0,0], [-8.54,   277.57, 1.0,0,90,0], # Left Turn
      [19.19, 302.95, 1.0,0,180,0], [-3.42, 141.02, 1.0,0,270,0], # Right Turns
      [16.98, 106.46, 1.0,0,180,0], [-7.82, 282.72, 1.0,0,90,0], #Two Turns  
    ]

    LANE_END_POINTS = [
      [148.41, 306.5, 1.0,0,0,0], [-8.54, 277.57, 1.0,0,90,0], #Straight
      [193.09, 277.68, 1.0,0,270,0], [18.61,  306.92, 1.0,0,0,0], # Left Turn
      [-3.16, 276.42, 1.0,0,270,0], [20.45,   109.42, 1.0,0,0,0], # Right Turns
      [13.05, 306.95, 1.0,0,0,0], [193.49, 287.6, 1.0,0,270,0], #Two Turns
    ]

    def __init__(self, config):
        # HUD
        self._pg_screen = None
        self._pg_font = None
        self._pg_scale = 4
        super().__init__(config)

        # ---- traffic density scheduling ----
        def _cfg_get(key, default=None):
            # embodied.Config supports both attribute + dict-like access depending on setup
            if hasattr(self._config, key):
                return getattr(self._config, key)
            try:
                return self._config.get(key, default)
            except Exception:
                return default

        self.traffic_densities = list(_cfg_get("traffic_densities", []))
        if not self.traffic_densities:
            # fallback to fixed behavior
            self.traffic_densities = [int(_cfg_get("num_vehicles", 0))]

        self.traffic_change_steps = int(_cfg_get("traffic_change_steps", 0) or 0)
        self.traffic_change_episodes = int(_cfg_get("traffic_change_episodes", 0) or 0)
        self.traffic_mode = str(_cfg_get("traffic_mode", "random")).lower()
        traffic_seed = int(_cfg_get("traffic_seed", 0) or 0)

        base_seed = int(_cfg_get("seed", 0) or 0)
        self._traffic_rng = random.Random(traffic_seed + base_seed)

        # ---- route-group sampling ----
        self.route_group = _cfg_get("route_group", None)  # e.g., "straight"
        self.route_groups = dict(_cfg_get("route_groups", {}) or {})
        self.route_mode = str(_cfg_get("route_mode", "random")).lower()
        route_seed = int(_cfg_get("route_seed", 0) or 0)
        self._route_rng = random.Random(route_seed + base_seed)
        self._route_cycle_idx = 0

        # Counters across the whole eval run
        self._total_steps_all = 0
        self._episodes_started = 0
        self._next_change_step = self.traffic_change_steps if self.traffic_change_steps else None
        self._traffic_cycle_idx = 0

        # This is what you should use when spawning traffic and logging
        self.current_vehicle_density = int(_cfg_get("num_vehicles", 0) or 0)

        # Speed thresholds
        self.max_distance_from_center = float(self._config.get("max_distance", 5.0))
        self.min_speed_mps = float(self._config.get("min_speed_mps", 0.5))
        self.min_speed_timeout = float(self._config.get("min_speed_timeout", 10.0))
        self._fps = float(1.0 / float(self._config.world.get("fixed_delta_seconds", 0.1))) if hasattr(self._config, "world") else 10.0
        self.offroad_patience_steps = max(1, int(self._config.get("offroad_patience_s", 0.2) * self._fps))

        self.waypoints = []
        self._slow_steps = 0
        self._offroad_steps = 0
        self._hud_slow_steps = 0
        self.allowed_end_indices: List[int] = []

        # Spectator
        self.spectator_enable = self._config.get("spectator_enable", True)
        self.spectator_mode = self._config.get("spectator_mode", "chase")
        self.spectator_distance = float(self._config.get("spectator_distance", 8.0))
        self.spectator_height = float(self._config.get("spectator_height", 3.0))
        self.spectator_pitch = float(self._config.get("spectator_pitch", -10.0))

        # Route pooling
        self._init_route_routing()

        # --- comfort bookkeeping (evaluation) ---
        self._dt = 1.0 / float(self._fps) if getattr(self, "_fps", 0) else 0.1
        self._prev_speed_ms = None
        self._prev_acc_ms2 = 0.0
        self._prev_yaw_deg = None
        self._prev_control = None
        self._prev_lat_acc_ms2 = 0.0

        self.current_vehicle_density = int(getattr(self._config, "num_vehicles", self._config.get("num_vehicles", 0)))

    def _sample_lane_pair_index(self) -> int:
        """
        Choose a lane_pair_index for the next episode.
        - If route_group is set (e.g., 'straight'), sample from route_groups[route_group]
        - Else sample from all available indices (fallback)
        """
        # Normalize null-ish values
        rg = self.route_group
        if isinstance(rg, str) and rg.lower() in ("none", "null", ""):
            rg = None

        if rg is not None:
            if rg not in self.route_groups:
                raise KeyError(f"route_group='{rg}' not in route_groups={list(self.route_groups.keys())}")
            candidates = list(self.route_groups[rg])
        else:
            # fallback: all indices
            candidates = list(range(len(self.LANE_START_POINTS)))

        if not candidates:
            raise ValueError(f"No route candidates for route_group={rg}")

        if self.route_mode == "cycle":
            idx = candidates[self._route_cycle_idx % len(candidates)]
            self._route_cycle_idx += 1
            return int(idx)
        else:
            return int(self._route_rng.choice(candidates))


    def _resample_traffic_density_if_needed(self, force: bool = False) -> None:
        if not self.traffic_densities:
            return

        do_change = force

        # Change every N episodes (episode-boundary only)
        if (not do_change) and self.traffic_change_episodes and self._episodes_started > 0:
            if (self._episodes_started % self.traffic_change_episodes) == 0:
                do_change = True

        # Change when crossing step thresholds (applied at next reset)
        if (not do_change) and self.traffic_change_steps and (self._next_change_step is not None):
            if self._total_steps_all >= self._next_change_step:
                do_change = True

        if not do_change:
            return

        if self.traffic_mode == "cycle":
            dens = self.traffic_densities[self._traffic_cycle_idx % len(self.traffic_densities)]
            self._traffic_cycle_idx += 1
        else:
            dens = self._traffic_rng.choice(self.traffic_densities)

        self.current_vehicle_density = int(dens)

        if self.traffic_change_steps:
            self._next_change_step = self._total_steps_all + self.traffic_change_steps

    def _ensure_pygame(self, h, w):
        if self._pg_screen is None:
            pygame.init()
            self._pg_screen = pygame.display.set_mode((int(w*self._pg_scale), int(h*self._pg_scale)))
            pygame.display.set_caption("CarlaLaneFollowingEnv")
            self._pg_font = pygame.font.SysFont("monospace", 18)

    def _extract_frame(self, obs):
        for k in ("camera","semantic_segmentation","birdeye_wpt","birdeye_with_traffic_lights"):
            if k in obs:
                frame = obs[k]; break
        else:
            return None
        if frame.ndim == 4: frame = frame[0]
        if frame.dtype != np.uint8:
            frame = np.clip(frame * (255.0 if frame.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return frame

    def _render_pygame(self, obs, info):
        frame = self._extract_frame(obs)
        if frame is None: return
        h, w = frame.shape[:2]
        self._ensure_pygame(h, w)
        surf = pygame.surfarray.make_surface(frame.transpose(1,0,2))
        surf = pygame.transform.smoothscale(surf, (int(w*self._pg_scale), int(h*self._pg_scale)))
        self._pg_screen.blit(surf, (0,0))
        sh = sb.get()
        lines = [
            f"t={self._time_step}", f"env_step={self._env_step}",
            f"stage={self.stage}", f"town={getattr(self,'town','NA')}",
            f"route_idx={getattr(self,'_lane_pair_index','NA')}",
            f"speed_mps={info.get('speed_norm',0.0):.2f}",
            f"dest={int(info.get('destination_reached',False))}",
            f"coll={int(info.get('is_collision',False))}",
            f"shield={'OVERRIDE' if sh.unsafe else 'OK'} ",
            f"{sh.orig_idx}->{sh.idx}  ",
            f"risk:{sh.orig_risk:.2f}->{sh.chosen_risk:.2f}  ",
            f"τ/λ:{sh.lam:.2f}",
        ]

        


        pad=6; line_h=20
        maxw = max(self._pg_font.size(s)[0] for s in lines) if lines else 0
        rect = pygame.Surface((maxw+2*pad, len(lines)*line_h+2*pad), pygame.SRCALPHA)
        rect.fill((0,0,0,160)); self._pg_screen.blit(rect,(8-pad,8-pad))
        for i,s in enumerate(lines):
            img = self._pg_font.render(s, True, (255,255,255))
            self._pg_screen.blit(img, (8, 8+i*line_h))
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
        pygame.display.flip()

    def _update_spectator(self):
        if not self.spectator_enable or not hasattr(self, "ego"): return
        spec = self._world.carla_world.get_spectator()
        t = self.ego.get_transform()
        if self.spectator_mode == "topdown":
            loc = carla.Location(x=t.location.x, y=t.location.y, z=t.location.z + max(self.spectator_height, 15.0))
            rot = carla.Rotation(pitch=-90.0, yaw=t.rotation.yaw, roll=0.0)
        else:
            fwd = t.get_forward_vector()
            loc = carla.Location(x=t.location.x - fwd.x * self.spectator_distance,
                                 y=t.location.y - fwd.y * self.spectator_distance,
                                 z=t.location.z + self.spectator_height)
            rot = carla.Rotation(pitch=self.spectator_pitch, yaw=t.rotation.yaw, roll=0.0)
        spec.set_transform(carla.Transform(loc, rot))

    def is_off_road(self) -> bool:
        ego_loc = self.ego.get_location()
        wp_any = self._world._map.get_waypoint(ego_loc, project_to_road=False)
        if wp_any is None or not (wp_any.lane_type & carla.LaneType.Driving):
            return True
        wp_drv = self._world._map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp_drv is None: return True
        r = wp_drv.transform.get_right_vector()
        dx = ego_loc.x - wp_drv.transform.location.x
        dy = ego_loc.y - wp_drv.transform.location.y
        cte = abs(dx * r.x + dy * r.y)
        return cte > self.max_distance_from_center

    # ---- lifecycle ----
    def on_reset(self):
        self._time_step = 0
        self._env_step = 0
        self._slow_steps = 0
        self._offroad_steps = 0
        self._hud_slow_steps = 0

        # self.ego = self._world.spawn_actor()
        # self._select_lane_pair(index=self._config.get("lane_pair_index", None))
        # pick a route index for this episode
        route_idx = self._sample_lane_pair_index()
        self._select_lane_pair(index=route_idx)
        assert getattr(self, "_lane_start_transform", None) is not None, "Start transform not set"

        self.ego = self._world.spawn_actor(transform=self._lane_start_transform)

        print("goal: " , self.goal)

        # if getattr(self, "_lane_start_transform", None):
        #     self.ego.set_transform(self._lane_start_transform)

        self._update_spectator()

        # self._world.spawn_auto_actors(self._config.num_vehicles)

        # resample on episode boundaries
        self._resample_traffic_density_if_needed(force=(self._episodes_started == 0))

        self._world.spawn_auto_actors(self.current_vehicle_density)

        # count episodes started
        self._episodes_started += 1

        # --- reset comfort state ---
        self._dt = 1.0 / float(getattr(self, "_fps", 10.0))
        self._prev_speed_ms = 0.0
        self._prev_acc_ms2 = 0.0
        try:
            self._prev_yaw_deg = float(self.ego.get_transform().rotation.yaw)
        except Exception:
            self._prev_yaw_deg = None
        self._prev_control = None
        self._prev_lat_acc_ms2 = 0.0


        self.ego_planner = RandomPlanner(vehicle=self.ego)
        self.on_step()  # compute initial waypoints
        self.obs, _ = self._observer.get_observation(self.get_state())

    def step(self, action):
        # world.step and core bookkeeping are in base
        obs, rew, done, info = super().step(action)

        # basic kinematics for HUD and terminals
        vx, vy = self.ego.get_velocity().x, self.ego.get_velocity().y
        speed_ms = math.hypot(vx, vy)

        dt = float(getattr(self, "_dt", 0.1))
        if dt <= 0:
            dt = 0.1

        prev_speed = speed_ms if self._prev_speed_ms is None else float(self._prev_speed_ms)
        acc_ms2 = float((speed_ms - prev_speed) / dt)
        jerk_ms3 = float((acc_ms2 - float(getattr(self, "_prev_acc_ms2", 0.0))) / dt)

        yaw_rate_rps = 0.0
        lat_acc_ms2 = 0.0
        try:
            yaw_deg = float(self.ego.get_transform().rotation.yaw)
            if self._prev_yaw_deg is not None:
                dyaw = yaw_deg - float(self._prev_yaw_deg)
                dyaw = (dyaw + 180.0) % 360.0 - 180.0
                yaw_rate_rps = math.radians(dyaw) / dt
                lat_acc_ms2 = float(speed_ms * yaw_rate_rps)
            self._prev_yaw_deg = yaw_deg
        except Exception:
            pass

        prev_lat_acc = float(getattr(self, "_prev_lat_acc_ms2", 0.0))
        lat_jerk_ms3 = float((lat_acc_ms2 - prev_lat_acc) / dt)
        self._prev_lat_acc_ms2 = float(lat_acc_ms2)

        self._prev_speed_ms = float(speed_ms)
        self._prev_acc_ms2 = float(acc_ms2)

        # control deltas: use CARLA-reported control if available
        dsteer = dthrottle = dbrake = 0.0
        try:
            ctrl = self.ego.get_control()
            if self._prev_control is not None:
                dsteer = float(abs(ctrl.steer - self._prev_control.steer))
                dthrottle = float(abs(ctrl.throttle - self._prev_control.throttle))
                dbrake = float(abs(ctrl.brake - self._prev_control.brake))
            self._prev_control = ctrl
        except Exception:
            pass

        info["speed_ms"] = float(speed_ms)
        info["comfort_acc_ms2"] = float(acc_ms2)
        info["comfort_jerk_ms3"] = float(jerk_ms3)
        info["comfort_yaw_rate_rps"] = float(yaw_rate_rps)
        info["comfort_lat_acc_ms2"] = float(lat_acc_ms2)
        info["comfort_dsteer_abs"] = float(dsteer)
        info["comfort_dthrottle_abs"] = float(dthrottle)
        info["comfort_dbrake_abs"] = float(dbrake)
        info["traffic_density"] = int(getattr(self, "current_vehicle_density", -1))

        # ensure these get forwarded through the logging channel
        info["log_comfort_acc_ms2"] = float(acc_ms2)
        info["log_comfort_jerk_ms3"] = float(jerk_ms3)
        info["log_comfort_lat_acc_ms2"] = float(lat_acc_ms2)
        info["log_comfort_dsteer_abs"] = float(dsteer)
        info["log_comfort_dthrottle_abs"] = float(dthrottle)

        info["log_speed_ms"] = float(speed_ms)
        info["log_traffic_density"] = int(getattr(self, "current_vehicle_density", -1))
        info["log_lane_pair_index"] = int(getattr(self, "_lane_pair_index", -1))

        # self._slow_steps = self._slow_steps + 1 if speed_ms < self.min_speed_mps else 0
        self._hud_slow_steps = self._hud_slow_steps + 1 if speed_ms < self.min_speed_mps else 0
        info["hud_slow_steps"] = int(self._hud_slow_steps)
        self._offroad_steps = self._offroad_steps + 1 if self.is_off_road() else 0

        info["speed_norm"] = speed_ms

        self._total_steps_all += 1

        info["traffic_density"] = int(getattr(self, "current_vehicle_density", -1))
        info["lane_pair_index"] = int(getattr(self, "_lane_pair_index", -1))

        info["comfort_lat_jerk_ms3"] = float(lat_jerk_ms3)
        info["log_comfort_lat_jerk_ms3"] = float(lat_jerk_ms3)

        return obs, rew, done, info
