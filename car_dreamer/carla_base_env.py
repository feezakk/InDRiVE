from abc import abstractmethod
from typing import Dict, Tuple

import carla
import gym
import numpy as np
from gym import spaces

from .toolkit import Observer, WorldManager

import csv, os, time


class CarlaBaseEnv(gym.Env):
    def __init__(self, config):
        self._config = config
        self._world = WorldManager(self._config)
        self._world.on_reset(self.on_reset)
        self._world.on_step(self.on_step)
        self._observer = Observer(self._world, self._config.observation)

        # Stage and seed (used by children for routing/determinism)
        self.stage = str(self._config.get("stage", "train"))
        self.seed = int(self._config.get("seed", 0))

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self._time_step = 0
        self._env_step = 0

        # ---- episode aggregates ----
        self._episode_idx = 0
        self._ep = dict(
            lat_sum=0.0, lat_cnt=0, lat_max=0.0,
            invasions=0,                  # lane invasion events
            ttc_min=np.inf,               # min TTC seen
            ret_sum=0.0,                  # episode return
        )

        # OPTIONAL: episode‑level CSV (one row per episode)
        ecfg = getattr(self._config, "episode_csv", {}) if hasattr(self._config, "episode_csv") else self._config.get("episode_csv", {})
        self._epcsv_enable = bool(getattr(ecfg, "enable", False) if hasattr(ecfg, "enable") else ecfg.get("enable", False))
        self._epcsv_writer = None
        if self._epcsv_enable:
            epath = getattr(ecfg, "path", None) if hasattr(ecfg, "path") else ecfg.get("path")
            if not epath:
                epath = f"episodes_{os.getpid()}_{int(time.time())}.csv"
            os.makedirs(os.path.dirname(epath) or ".", exist_ok=True)
            self._epcsv_file = open(epath, "w", newline="")
            self._epcsv_writer = csv.DictWriter(
                self._epcsv_file,
                fieldnames=["episode","steps","ep_return","ep_lat_err_mean","ep_lat_err_max","ep_lane_invasions","ep_min_ttc"]
            )
            self._epcsv_writer.writeheader()


        # --- per-step CSV logging setup ---
        self._csv_file = None
        self._csv_writer = None
        # enable via config; default off
        csv_cfg = getattr(self._config, "perstep_csv", {}) if hasattr(self._config, "perstep_csv") else self._config.get("perstep_csv", {})
        self._csv_enable = bool(getattr(csv_cfg, "enable", False) if hasattr(csv_cfg, "enable") else csv_cfg.get("enable", True))
        if self._csv_enable:
            path = getattr(csv_cfg, "path", None) if hasattr(csv_cfg, "path") else csv_cfg.get("path")
            if not path:
                path = f"perstep_{os.getpid()}_{int(time.time())}.csv"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._csv_file = open(path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=[
                    "env_step","episode_step","reward","done",
                    "r_waypoints", "r_speed", "r_collision", "r_out_of_lane", "r_destination", "time_penalty", "r_progress", "r_overspeed", "r_ttc",
                    "is_collision","out_of_lane", "is_off_road","is_wrong_direction","too_slow","time_exceeded","is_destination_reached",
                    "speed_kmh","wpt_dis","ttc", "lat_err", "lane_width"
                ],
            )
            self._csv_writer.writeheader()

    @abstractmethod
    def on_reset(self) -> None:
        pass

    @abstractmethod
    def apply_control(self, action) -> None:
        pass

    @abstractmethod
    def on_step(self) -> None:
        pass

    @abstractmethod
    def reward(self) -> Tuple[float, Dict]:
        pass

    @abstractmethod
    def get_terminal_conditions(self) -> Dict[str, bool]:
        pass

    def get_ego_vehicle(self) -> carla.Actor:
        return self.ego

    def get_state(self) -> Dict:
        return getattr(self, "_state", {})

    def _get_action_space(self):
        action_config = self._config.action
        if action_config.discrete:
            self.n_steer = len(action_config.discrete_steer)
            self.n_acc = len(action_config.discrete_acc)
            return spaces.Discrete(self.n_steer * self.n_acc)
        else:
            return spaces.Box(
                low=np.array([action_config.continuous_acc[0], action_config.continuous_steer[0]]),
                high=np.array([action_config.continuous_acc[1], action_config.continuous_steer[1]]),
                dtype=np.float32,
            )

    def _get_observation_space(self):
        return self._observer.get_observation_space()

    # Safe stubs; children can override
    def _update_spectator(self): 
        pass

    def _render_pygame(self, obs, info):
        pass

    def reset(self):
        print("[CARLA] Reset environment")
        self._observer.destroy()
        self._world.reset()
        self._observer.reset(self.get_ego_vehicle())
        self._update_spectator()
        self._time_step = 0
        print("[CARLA] Environment reset")
        self.obs, _ = self._observer.get_observation(self.get_state())
        return self.obs

    def get_vehicle_control(self, action):
        action_config = self._config.action
        if action_config.discrete:
            acc = action_config.discrete_acc[action // self.n_steer]
            steer = action_config.discrete_steer[action % self.n_steer]
        else:
            acc, steer = action[0], action[1]
        if acc > 0:
            throttle, brake = np.clip(acc, 0, 1), 0
        else:
            throttle, brake = 0, np.clip(-acc, 0, 1)
        return carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))

    def _is_terminal(self):
        terminal_conds = self.get_terminal_conditions()
        terminal = any(bool(v) for v in terminal_conds.values())
        for k, v in terminal_conds.items():
            if v:
                print(f"[CARLA] Terminal condition triggered: {k}")
            terminal_conds[k] = np.array([v], dtype=np.bool_)
        if terminal:
            terminal_conds["episode_timesteps"] = self._time_step
        terminal_conds["terminal"] = terminal
        return terminal, terminal_conds

    def step(self, action):
        self.apply_control(action)
        self._world.step()
        self._time_step += 1
        self._env_step += 1
        self._update_spectator()

        env_state = self.get_state()
        is_terminal, terminal_conds = self._is_terminal()
        self.obs, obs_info = self._observer.get_observation(env_state)
        reward, reward_info = self.reward()

        info = {**env_state, **terminal_conds, **obs_info, **reward_info, "action": action}

        # write one CSV row per env step (includes lat_err, lane_width)
        # self._log_csv_step(reward, is_terminal, info)

        if self._config.eval:
            info = {f"eval_{k}": v for k, v in info.items()}
            self.obs = {**self.obs, **info}

        # ---- episode aggregates (no per‑step averaging) ----
        lat = info.get("lat_err", None)
        if lat is not None and np.isfinite(lat):
            self._ep["lat_sum"] += float(lat)
            self._ep["lat_cnt"] += 1
            self._ep["lat_max"] = max(self._ep["lat_max"], float(lat))
        if "ttc" in info and np.isfinite(info["ttc"]):
            self._ep["ttc_min"] = min(self._ep["ttc_min"], float(info["ttc"]))
        if info.get("lane_invasion", False):
            self._ep["invasions"] += 1
        self._ep["ret_sum"] += float(reward)

        # ---- on terminal: attach episode summary to info (and optionally CSV) ----
        if is_terminal:
            mean_lat = (self._ep["lat_sum"] / max(1, self._ep["lat_cnt"])) if self._ep["lat_cnt"] else None
            min_ttc = None if not np.isfinite(self._ep["ttc_min"]) else float(self._ep["ttc_min"])
            ep_summary = {
                "ep_index": self._episode_idx,
                "ep_steps": int(self._time_step),
                "ep_return": float(self._ep["ret_sum"]),
                "ep_lat_err_mean": None if mean_lat is None else float(mean_lat),
                "ep_lat_err_max": None if self._ep["lat_cnt"] == 0 else float(self._ep["lat_max"]),
                "ep_lane_invasions": int(self._ep["invasions"]),
                "ep_min_ttc": min_ttc,
            }
            info.update(ep_summary)

            if self._epcsv_writer:
                self._epcsv_writer.writerow({
                    "episode": ep_summary["ep_index"],
                    "steps": ep_summary["ep_steps"],
                    "ep_return": ep_summary["ep_return"],
                    "ep_lat_err_mean": ep_summary["ep_lat_err_mean"],
                    "ep_lat_err_max": ep_summary["ep_lat_err_max"],
                    "ep_lane_invasions": ep_summary["ep_lane_invasions"],
                    "ep_min_ttc": ep_summary["ep_min_ttc"],
                })
                self._epcsv_file.flush()

            # reset accumulators for next episode
            self._episode_idx += 1
            self._ep = dict(lat_sum=0.0, lat_cnt=0, lat_max=0.0, invasions=0, ttc_min=np.inf, ret_sum=0.0)

        

        # Optional on‑screen debug
        try:
            if getattr(self._config.display, "enable", False):
                self._render_pygame(self.obs, info)
        except Exception:
            pass

        return (self.obs, reward, is_terminal, info)

    def is_collision(self):
        return bool(self.obs.get("collision", np.array([0]))[0])

    def _render(self, obs, info):
        pass

    def _csv_scalar(self, v, default=0.0):
        try:
            a = np.asarray(v)
            if a.size == 0:
                return float(default)
            return float(a.reshape(-1)[0])
        except Exception:
            return float(default)

    def _log_csv_step(self, reward, done, info):
        if not self._csv_writer:
            return
        

        fieldnames=[
            "env_step","episode_step","reward","done",
            "r_waypoints", "r_speed", "r_collision", "r_out_of_lane", "r_destination", "time_penalty", "r_progress", "r_overspeed", "r_ttc",
            "is_collision","out_of_lane", "is_off_road","is_wrong_direction","too_slow","time_exceeded","is_destination_reached",
            "speed_kmh","wpt_dis","ttc", "lat_err", "lane_width"
        ],
        row = dict(
            env_step=int(self._env_step),
            episode_step=int(self._time_step),
            reward=float(self._csv_scalar(reward, 0.0)),
            done=int(bool(done)),

            r_waypoints=float(self._csv_scalar(info.get("r_waypoints", 0.0), 0.0)),
            r_speed=float(self._csv_scalar(info.get("r_speed", 0.0), 0.0)),
            r_collision=float(self._csv_scalar(info.get("r_collision", 0.0), 0.0)),
            r_out_of_lane=float(self._csv_scalar(info.get("r_out_of_lane", 0.0), 0.0)),
            r_destination=float(self._csv_scalar(info.get("r_destination", 0.0), 0.0)),
            time_penalty=float(self._csv_scalar(info.get("time_penalty", 0.0), 0.0)),
            r_progress=float(self._csv_scalar(info.get("r_progress", 0.0), 0.0)),
            r_overspeed=float(self._csv_scalar(info.get("r_overspeed", 0.0), 0.0)),
            r_ttc=float(self._csv_scalar(info.get("r_ttc", 0.0), 0.0)),



            is_collision=int(bool(info.get("is_collision", 0))),
            is_off_road=int(bool(info.get("is_off_road", info.get("off_road", 0)))),
            too_slow=int(bool(info.get("too_slow", 0))),
            time_exceeded=int(bool(info.get("time_exceeded", 20000))),
            is_destination_reached=int(bool(info.get("is_destination_reached", 0))),
            out_of_lane=int(bool(info.get("out_of_lane", 0))),
            
            # speed_kmh=float(self._csv_scalar(info.get("speed_norm", 0.0), 0.0)),
            speed_kmh=float(self._csv_scalar(info.get("speed_norm", 0.0) * 3.6, 0.0)),
            wpt_dis=float(self._csv_scalar(info.get("wpt_dis", 0.0), 0.0)),
            ttc=float(self._csv_scalar(info.get("ttc", np.nan), np.nan)),

            lat_err=float(self._csv_scalar(info.get("lat_err", np.nan), np.nan)),
            lane_width=float(self._csv_scalar(info.get("lane_width", np.nan), np.nan)),
        )
        self._csv_writer.writerow(row)
        self._csv_file.flush()

