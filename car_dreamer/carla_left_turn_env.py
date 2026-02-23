# from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedEndingPlanner, get_vehicle_pos

import pygame
import numpy as np
import carla


class CarlaLeftTurnEnv(CarlaWptEnv):
    
    def on_reset(self) -> None:
        sp = np.random.randint(0, len(self._config.lane_start_points) - 1)
        self.ego_src = self._config.lane_start_points[sp]
        ego_transform = carla.Transform(
            carla.Location(x=self.ego_src[0], y=self.ego_src[1], z=self.ego_src[2])
        )
        self.ego = self._world.spawn_actor(transform=ego_transform)

        # Path planning
        self.ego_dest = self._config.lane_end_points[sp]
        print(self.ego_dest)
        dest_location = carla.Location(x=self.ego_dest[0], y=self.ego_dest[1], z=self.ego_dest[2])
        self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]
        # print("[reset]",len(self.waypoints))

        # print("[reset] planner stat", self.planner_stats)

        self._update_spectator()

    def on_step(self) -> None:

        # self._pg_screen = None; self._pg_font = None; self._pg_scale = 4
        self._pg_scale = getattr(self, "_pg_scale", 4)
        self._env_step = getattr(self, "_env_step", 0)
        self._env_step = getattr(self, "_env_step", 0)


        # spectator view config
        self.spectator_enable = self._config.get("spectator_enable", True)
        self.spectator_mode = self._config.get("spectator_mode", "chase")   # "chase" or "topdown"
        self.spectator_distance = self._config.get("spectator_distance", 8.0)
        self.spectator_height = self._config.get("spectator_height", 3.0)
        self.spectator_pitch = self._config.get("spectator_pitch", -10.0)

        # print("[step]", len(self.waypoints))
        # print("[step] planner stat", self.planner_stats)

        super().on_step()

    def _ensure_pygame(self, _h, _w):
        import pygame
        target = (512, 512)
        flags = pygame.DOUBLEBUF  # exact size; no SCALED
        if getattr(self, "_pg_screen", None) is None:
            pygame.init()
            try:
                self._pg_screen = pygame.display.set_mode(target, flags, vsync=1)
            except TypeError:
                self._pg_screen = pygame.display.set_mode(target, flags)
            pygame.display.set_caption("CarlaLaneFollowingEnv 512x512")
            self._pg_font = pygame.font.SysFont("monospace", 18)
            self._pg_clock = pygame.time.Clock()
        elif self._pg_screen.get_size() != target:
            self._pg_screen = pygame.display.set_mode(target, flags)

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

    def _blit_text(self, surface, lines, x=8, y=8, line_h=20):
        # translucent panel
        pad = 6
        maxw = max(self._pg_font.size(s)[0] for s in lines) if lines else 0
        rect = pygame.Surface((maxw + 2*pad, len(lines)*line_h + 2*pad), pygame.SRCALPHA)
        rect.fill((0, 0, 0, 160))
        surface.blit(rect, (x- pad, y- pad))
        # text
        for i, s in enumerate(lines):
            img = self._pg_font.render(s, True, (255, 255, 255))
            surface.blit(img, (x, y + i*line_h))

    def _render_pygame(self, obs, info):
        frame = self._extract_frame(obs)
        if frame is None:
            return
        h, w = frame.shape[:2]
        self._ensure_pygame(h, w)

        # Pygame expects WxH, so transpose
        self._pg_screen.fill((0, 0, 0))  # avoid residual artifacts
        surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        # surf = pygame.transform.smoothscale(surf, (w*self._pg_scale, h*self._pg_scale))
        surf = pygame.transform.smoothscale(surf, self._pg_screen.get_size())

        self._pg_screen.blit(surf, (0, 0))

        # overlay lines (add more as needed)
        action_idx = None

        if "action" in info:
            a = np.asarray(info["action"]).reshape(-1)
            action_idx = int(np.argmax(a))

        lines = [
            f"t={self._time_step}",
            f"env_step={self._env_step}",
            f"lane_start_idx={self.ego_src}",
            f"end_idxs={self.ego_dest}",
            f"speed_parallel={info.get('speed_parallel', 0):.2f}",
            f"speed_perpendicular={info.get('speed_perpendicular', 0):.2f}",
            f"r_waypoints={info.get('r_waypoints', 0):.3f}",
            f"r_speed={info.get('r_speed', 0):.3f}",
            f"r_collision={info.get('r_collision', 0):.3f}",
            f"r_out_of_lane={info.get('r_out_of_lane', 0):.3f}",
            f"time_penalty={info.get('time_penalty', 0):.3f}",
            f"total_reward={info.get('total_reward', 0):.3f}",
            f"dest_reached={int(info.get('destination_reached', 0))}",
            f"action_idx={action_idx if action_idx is not None else 'NA'}",

            f"num_completed={info.get('num_completed', -1)}",
            f"wpt_dist={self.get_wpt_dist(np.array(get_vehicle_pos(self.get_ego_vehicle()))):.2f}",

            f"waypoints={len(self.waypoints)}",
        ]

        line_h = self._pg_font.get_linesize()
        pad = 14                      # same pad used in _blit_text
        base_y = 8 + len(lines)*line_h + 2*pad + 8  # +8 gap

        # self._blit_text(self._pg_screen, shield_lines, x=8, y=base_y)

        self._blit_text(self._pg_screen, lines)

        # keep window responsive
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        from dreamerv3 import shield_bus as sb
        s = sb.get()
        self._blit_text(
            self._pg_screen,
            [
                f"SHIELD: λ={getattr(s,'lam',0):.3f}  unsafe={int(s.unsafe)}",
                f"{getattr(s,'orig_idx',-1)}→{getattr(s,'idx',-1)}",
                f"risk {getattr(s,'orig_risk',0):.3f}→{getattr(s,'chosen_risk',0):.3f}",
            ],
            x=8, y=base_y
        )


        pygame.display.flip()
        if hasattr(self, "_pg_clock"):
            self._pg_clock.tick(60)

    # --- in CarlaBaseEnv ---------------------------------------------------------

    def _update_spectator(self) -> None:
        if not getattr(self, "spectator_enable", False):
            return
        if not hasattr(self, "_world") or not hasattr(self, "ego") or self.ego is None:
            return
        # get CARLA world and spectator
        world = getattr(self._world, "carla_world", None)
        if world is None:
            return
        spectator = world.get_spectator()
        t = self.get_ego_vehicle().get_transform()
        if getattr(self, "spectator_mode", "chase") == "topdown":
            loc = carla.Location(x=t.location.x, y=t.location.y,
                                z=t.location.z + max(getattr(self, "spectator_height", 15.0), 15.0))
            rot = carla.Rotation(pitch=-90.0, yaw=t.rotation.yaw, roll=0.0)
        else:
            fwd = t.get_forward_vector()
            dist = getattr(self, "spectator_distance", 8.0)
            height = getattr(self, "spectator_height", 3.0)
            pitch = getattr(self, "spectator_pitch", -10.0)
            loc = carla.Location(x=t.location.x - fwd.x * dist,
                                y=t.location.y - fwd.y * dist,
                                z=t.location.z + height)
            rot = carla.Rotation(pitch=pitch, yaw=t.rotation.yaw, roll=0.0)
        spectator.set_transform(carla.Transform(loc, rot))


        # if len(self.actor_flow) > 0:
        #     vehicle = self.actor_flow[0]
        #     x, y = get_vehicle_pos(self.actor_flow[0])
        #     if y > -99.4 or y < -171.4 or x > 57.6:
        #         self._world.destroy_actor(vehicle.id)
        #         self.actor_flow.popleft()
        
