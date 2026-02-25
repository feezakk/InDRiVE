import random
import carla
import numpy as np

class RoutePoolMixin:
    """Stage- and town-aware route selection with deterministic cycling."""

    def _init_route_routing(self):
        # World config
        wcfg = self._config.get("world", {}) if isinstance(self._config, dict) else getattr(self._config, "world", {})
        self.town = wcfg.get("town", "Town01")

        # Allow explicit flat list (--env.route_pool "0,2,4")
        rp = self._config.get("route_pool", None)
        if isinstance(rp, str):
            rp = [int(x) for x in rp.split(",") if x.strip()]
        self._route_pool_explicit = rp

        # Map form from YAML: route_pools: {Town01: {pretrain:[...], finetune:[...]}, Town02:{...}}
        self._route_pools_map = getattr(self._config, "route_pools", {})

        # Load lane tables from config if provided, else class-level constants
        self._starts = self._config.get("lane_start_points", None) or getattr(self, "LANE_START_POINTS", [])
        self._ends   = self._config.get("lane_end_points",   None) or getattr(self, "LANE_END_POINTS", [])
        if not self._starts or not self._ends or len(self._starts) != len(self._ends):
            raise ValueError("lane_start_points and lane_end_points must exist and have equal length.")

        # Deterministic order per seed
        self._route_order = None
        self._route_cursor = 0

    def _active_route_pool(self):
        n = len(self._starts)
        # 1) explicit list wins
        if self._route_pool_explicit is not None:
            return [int(i) % n for i in self._route_pool_explicit]
        # 2) map by town/stage
        if self._route_pools_map and self.town in self._route_pools_map:
            stage_map = self._route_pools_map[self.town]
            if self.stage in stage_map:
                return [int(i) % n for i in stage_map[self.stage]]
        # 3) default to all routes
        return list(range(n))

    def _ensure_route_order(self):
        if self._route_order is None:
            pool = list(dict.fromkeys(self._active_route_pool()))  # unique, in order
            rng = random.Random(self.seed)
            rng.shuffle(pool)
            if not pool:
                raise ValueError("Active route_pool resolved to empty list.")
            self._route_order = pool
            self._route_cursor = 0

    def _select_lane_pair(self, index: int = None) -> None:
        self._ensure_route_order()
        n = len(self._starts)
        if index is not None:
            idx = int(index) % n
        else:
            idx = self._route_order[self._route_cursor % len(self._route_order)]
            self._route_cursor += 1

        # Set start and goal(s)
        self._lane_pair_index = idx
        if len(self._starts[idx]) == 3:
            sx, sy, _ = self._starts[idx]
            ex, ey, _ = self._ends[idx]
            self.lane_start_point = (float(sx), float(sy))
            self.goal = [(float(ex), float(ey))]
            start_loc = carla.Location(x=self.lane_start_point[0], y=self.lane_start_point[1], z=0.0)

            wp = self._world._map.get_waypoint(
                start_loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if wp is None:
                raise RuntimeError(f"No driving waypoint near start_loc={start_loc} for idx={idx}")

            start_tf = wp.transform
            start_tf.location.z = max(start_tf.location.z, 0.5)

        else:
            print(self._starts[idx])
            print(self._ends[idx])
            sx, sy, sz, sq, sw, se = self._starts[idx]
            ex, ey, ez, eq, ew, ee = self._ends[idx]
            self.lane_start_point = (float(sx), float(sy), float(sz), float(sq), float(sw), float(se))
            self.goal = [(float(ex), float(ey), float(ez), float(eq), float(ew), float(ee))]
            print("lane start point " , self.lane_start_point)
            start_loc = carla.Location(x = self.lane_start_point[0], y = self.lane_start_point[1], z = self.lane_start_point[2])
            print("start_loc " , start_loc)
            start_rot = carla.Rotation(yaw=float(sw), pitch=float(sq), roll=float(se))
            print("start rot " , start_rot)
            start_tf  = carla.Transform(start_loc, start_rot)

            # wp = self._world._map.get_waypoint(start_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            # tf = wp.transform
            # tf.location.z = max(tf.location.z, 0.5)
            # tf.rotation = start_rot  # keep your orientation instead of lane heading

        self._lane_start_transform = start_tf

        print("transform: " , self._lane_start_transform)

        print(f"[ROUTE] town={self.town} stage={self.stage} idx={idx} pool={self._active_route_pool()}")
