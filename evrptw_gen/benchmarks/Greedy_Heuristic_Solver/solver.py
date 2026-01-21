import heapq
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


def reconstruct_path(nxt: Dict[int, Dict[int, Optional[int]]], u: int, v: int) -> List[int]:
    """Return node sequence [u, ..., v] using next-hop table; empty if unreachable."""
    if u not in nxt or v not in nxt[u] or nxt[u][v] is None:
        return []
    path = [u]
    cur = u
    seen = {u}
    while cur != v:
        cur = nxt[cur][v]
        if cur is None or cur in seen:
            return []
        seen.add(cur)
        path.append(cur)
    return path


@dataclass
class PathInfo:
    arrival_node: int          # customer index (0-based)
    arrival_load: float
    arrival_time: float        # absolute time AFTER service at customer
    arrival_soc: float         # SoC when arriving at customer (before service)
    path_nodes: List[int]      # global node sequence from start -> ... -> customer (inclusive)


class GreedySolver:
    """
    Nodes indexing (global):
      0                      : depot
      1..num_customers       : customers (global = 1 + cus_idx)
      1+num_customers..end   : charging stations (CS)
    """

    def __init__(self, instance: Dict[str, Any]):
        self.instance = instance

        customers = instance["customers"]
        depot = instance["depot"]
        css = instance["stations"]

        # global nodes
        self.customers = np.array([(c["x"], c["y"]) for c in customers], dtype=float)
        self.depot = np.array([[depot["x"], depot["y"]]], dtype=float)
        self.css = np.array([(cs["x"], cs["y"]) for cs in css], dtype=float)
        self.nodes = np.vstack([self.depot, self.customers, self.css])

        # optional: string ids
        self.id_strs = {i: node["id"] for i, node in enumerate([depot] + customers + css)}

        self.num_customers = len(customers)
        self.num_cs = len(css)
        self.num_nodes = len(self.nodes)

        # global indices for CS
        self.depot_node = 0
        self.cs_start = 1 + self.num_customers
        self.cs_end = self.cs_start + self.num_cs
        self.css_idx = list(range(self.cs_start, self.cs_end))

        # ===== vehicle params =====
        self.velocity = float(instance["vehicle"]["v"])          # km/h
        self.consume_rate = float(instance["vehicle"]["r"])      # kWh/km
        self.charging_power = 1.0 / float(instance["vehicle"]["g"])  # g: inverse charging power (h/kWh); charging power: (kwh/h)
        self.fuel_cap = float(instance["vehicle"]["Q"])          # kWh
        self.load_cap = float(instance["vehicle"]["C"])         # tons

        # ===== time settings =====
        self.working_start = float(instance["meta"]["working_startTime"])
        self.working_end = float(instance["meta"]["working_endTime"])
        self.instance_end_time = float(instance["meta"]["instance_endTime"])

        # customer data (0-based cus_idx)
        self.service_time = [float(customers[i]["service"]) for i in range(self.num_customers)]
        self.time_windows = [(float(customers[i]["ready"]), float(customers[i]["due"])) for i in range(self.num_customers)]
        self.demands = [float(customers[i]["demand"]) for i in range(self.num_customers)]

        # geometry
        self.distance_matrix = self._distance_matrix(self.nodes)  # km

        # stop-graph allpairs on {depot} U CS
        self.time, self.nxt = self._precompute_stopgraph_allpairs()

        # visited & state (for current vehicle)
        self.visited = [False] * self.num_customers
        self.state = {
            "id": self.depot_node,
            "time": self.working_start,
            "soc": self.fuel_cap,
            "load": 0.0,
        }

        self.last_move_path: List[int] = []

    # ------------------------ physics helpers ------------------------

    def _distance_matrix(self, nodes: np.ndarray) -> np.ndarray:
        return np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)

    def travel_time(self, u: int, v: int) -> float:
        return float(self.distance_matrix[u][v] / self.velocity)  # hours

    def travel_energy(self, u: int, v: int) -> float:
        return float(self.consume_rate * self.distance_matrix[u][v])  # kWh

    def is_cs(self, node: int) -> bool:
        return self.cs_start <= node < self.cs_end

    def is_customer(self, node: int) -> bool:
        return 1 <= node <= self.num_customers

    # ------------------------ stop-graph allpairs ------------------------

    def _precompute_stopgraph_allpairs(self) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, Optional[int]]]]:
        """
        All-pairs shortest time and path on stop-graph nodes = {depot} U CS. O(N^3)

        Edge rule (between stop nodes only):
          u->v feasible if travel_energy(u,v) <= fuel_cap
          cost = travel_time(u,v) + (travel_energy(u,v)/P if v is CS else 0)
          (no charging term if v is depot)
        """
        depot = self.depot_node
        cs_nodes = list(self.css_idx)
        nodes = [depot] + cs_nodes
        is_cs_set = set(cs_nodes)

        INF = float("inf")
        P = float(self.charging_power)
        tol = 1e-9

        dist: Dict[int, Dict[int, float]] = {u: {v: INF for v in nodes} for u in nodes}
        nxt: Dict[int, Dict[int, Optional[int]]] = {u: {v: None for v in nodes} for u in nodes}

        for u in nodes:
            dist[u][u] = 0.0
            nxt[u][u] = u

        # direct edges
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                e = self.travel_energy(u, v)
                if e <= self.fuel_cap + tol:
                    if P <= 0 and v in is_cs_set:
                        continue
                    t = self.travel_time(u, v)
                    w = t if v == depot else (t + e / P)
                    if w < dist[u][v]:
                        dist[u][v] = w
                        nxt[u][v] = v

        # Floyd–Warshall O(N^3)
        for k in nodes:
            dk = dist[k]
            for i in nodes:
                dik = dist[i][k]
                if dik >= INF / 2:
                    continue
                di = dist[i]
                ni = nxt[i]
                for j in nodes:
                    if dk[j] >= INF / 2:
                        continue
                    cand = dik + dk[j]
                    if cand + 1e-12 < di[j]:
                        di[j] = cand
                        ni[j] = ni[k]

        return dist, nxt

    def _stop_path(self, u: int, v: int) -> List[int]:
        return reconstruct_path(self.nxt, u, v)

    # ------------------------ return-to-depot feasibility ------------------------

    def _can_return_to_depot(self, cus_global: int, cur_time: float, cur_soc: float) -> bool:
        """
        After serving customer (global id) at cur_time with cur_soc, check if can return to depot.
        Rule:
          - direct customer->depot is allowed if energy <= cur_soc and arrival <= working_end
          - else: customer->CS (energy <= cur_soc), charge to full at that CS, then CS->depot via stop-graph
        """
        depot = self.depot_node
        INF = float("inf")
        P = float(self.charging_power)
        if P <= 0:
            return False

        # direct
        e = self.travel_energy(cus_global, depot)
        t = self.travel_time(cus_global, depot)
        if e <= cur_soc - 1e-9 and cur_time + t <= self.instance_end_time + 1e-9:
            return True

        # via one CS then stop-graph
        for cs in self.css_idx:
            e1 = self.travel_energy(cus_global, cs)
            if e1 > cur_soc + 1e-9:
                continue
            t1 = self.travel_time(cus_global, cs)
            if cur_time + t1 > self.instance_end_time + 1e-9:
                continue

            soc_at_cs = cur_soc - e1
            charge_to_full = (self.fuel_cap - soc_at_cs) / P
            t_cs_dep = self.time[cs].get(depot, INF)
            if t_cs_dep >= INF / 2:
                continue

            # cur time + travel to cs + charge to full + travel to depot <= instance_end_time
            if cur_time + t1 + charge_to_full + t_cs_dep <= self.instance_end_time + 1e-9:
                return True

        return False

    def _return_to_depot_path(self, cur_global: int, cur_time: float, cur_soc: float) -> Optional[Tuple[List[int], float]]:
        """
        Build a feasible path from current node to depot, returning (path_nodes, arrival_time_at_depot).
        """
        depot = self.depot_node
        INF = float("inf")
        P = float(self.charging_power)
        if P <= 0:
            return None

        if cur_global == depot:
            return [depot], cur_time

        # current is CS
        if self.is_cs(cur_global):
            p = self._stop_path(cur_global, depot)
            if not p:
                return None
            return p, cur_time + self.time[cur_global][depot]

        # must be customer
        if not self.is_customer(cur_global):
            return None

        # direct
        e = self.travel_energy(cur_global, depot)
        t = self.travel_time(cur_global, depot)
        if e <= cur_soc + 1e-9 and cur_time + t <= self.working_end + 1e-9:
            return [cur_global, depot], cur_time + t

        # via CS (choose best arrival time)
        best: Optional[Tuple[float, List[int]]] = None  # (arrive_time_depot, path_nodes)
        for cs in self.css_idx:
            e1 = self.travel_energy(cur_global, cs)
            if e1 > cur_soc + 1e-9:
                continue
            t1 = self.travel_time(cur_global, cs)

            soc_at_cs = cur_soc - e1
            charge_to_full = (self.fuel_cap - soc_at_cs) / P

            t_cs_dep = self.time[cs].get(depot, INF)
            if t_cs_dep >= INF / 2:
                continue

            arrive_time_depot = cur_time + t1 + charge_to_full + t_cs_dep
            if arrive_time_depot > self.working_end + 1e-9:
                continue

            p_mid = self._stop_path(cs, depot)
            if not p_mid:
                continue

            path_nodes = [cur_global, cs] + p_mid[1:]
            if best is None or arrive_time_depot < best[0] - 1e-12:
                best = (arrive_time_depot, path_nodes)

        if best is None:
            return None
        return best[1], best[0]

    # ------------------------ shortest path: cur -> customer (direct + multi-hop via CS) ------------------------

    def shortest_time_cur_to_customer(self, cur_global: int, cus_idx: int) -> Optional[PathInfo]:
        # load feasibility
        cur_load = float(self.state["load"])
        if cur_load + self.demands[cus_idx] > self.load_cap + 1e-9:
            return None

        cur_time = float(self.state["time"])
        cur_soc = float(self.state["soc"])

        cus_global = 1 + cus_idx
        ready, due = self.time_windows[cus_idx]
        service_t = self.service_time[cus_idx]

        INF = float("inf")
        P = float(self.charging_power)
        if P <= 0:
            return None

        # ---------- direct ----------
        e_direct = self.travel_energy(cur_global, cus_global)
        t_direct = self.travel_time(cur_global, cus_global)
        arrival_raw = cur_time + t_direct  # physically arrive (before waiting)

        # your assumption: if arrival_raw <= due, service is allowed (finish may exceed due)
        if e_direct <= cur_soc + 1e-9 and arrival_raw <= due + 1e-9:
            start_service = max(arrival_raw, ready)
            finish_time = start_service + service_t
            arrive_soc = cur_soc - e_direct

            if self._can_return_to_depot(cus_global, finish_time, arrive_soc):
                return PathInfo(
                    arrival_node=cus_idx,
                    arrival_load=cur_load + self.demands[cus_idx],
                    arrival_time=finish_time,
                    arrival_soc=arrive_soc,
                    path_nodes=[cur_global, cus_global],
                )

        # ---------- multi-hop via CS ----------
        best: Optional[Tuple[float, float, List[int]]] = None
        # best = (finish_time, arrive_soc, path_nodes)

        def try_candidate(travel_total_time: float, arrive_soc2: float, path_nodes2: List[int]) -> None:
            nonlocal best
            arrival_raw2 = cur_time + travel_total_time
            if arrival_raw2 > due + 1e-9:
                return
            start_service2 = max(arrival_raw2, ready)
            finish_time2 = start_service2 + service_t
            if not self._can_return_to_depot(cus_global, finish_time2, arrive_soc2):
                return
            if best is None or finish_time2 < best[0] - 1e-12:
                best = (finish_time2, arrive_soc2, path_nodes2)

        # Case A: start is depot
        if cur_global == self.depot_node:
            depot = self.depot_node
            for cs_last in self.css_idx:
                mid = self.time[depot].get(cs_last, INF)
                if mid >= INF / 2:
                    continue

                e_last = self.travel_energy(cs_last, cus_global)
                if e_last > self.fuel_cap + 1e-9:
                    continue
                t_last = self.travel_time(cs_last, cus_global)

                travel_total = mid + t_last
                arrive_soc2 = self.fuel_cap - e_last

                p_mid = self._stop_path(depot, cs_last)

                if not p_mid:
                    continue
                path_nodes2 = p_mid + [cus_global]

                try_candidate(travel_total, arrive_soc2, path_nodes2)

        # Case B: start is a customer
        elif self.is_customer(cur_global):
            for cs_enter in self.css_idx:
                e1 = self.travel_energy(cur_global, cs_enter)
                if e1 > cur_soc + 1e-9:
                    continue
                t1 = self.travel_time(cur_global, cs_enter)

                soc_at_cs = cur_soc - e1
                charge_to_full = (self.fuel_cap - soc_at_cs) / P  # because start customer is not guaranteed full

                for cs_last in self.css_idx:
                    mid = self.time[cs_enter].get(cs_last, INF)
                    if mid >= INF / 2:
                        continue

                    e_last = self.travel_energy(cs_last, cus_global)
                    if e_last > self.fuel_cap + 1e-9:
                        continue
                    t_last = self.travel_time(cs_last, cus_global)

                    travel_total = t1 + charge_to_full + mid + t_last
                    arrive_soc2 = self.fuel_cap - e_last

                    p_mid = self._stop_path(cs_enter, cs_last)
                    if not p_mid:
                        continue
                    path_nodes2 = [cur_global, cs_enter] + p_mid[1:] + [cus_global]

                    try_candidate(travel_total, arrive_soc2, path_nodes2)

        else:
            return None

        if best is None:
            return None

        finish_time, arrive_soc2, path_nodes2 = best
        return PathInfo(
            arrival_node=cus_idx,
            arrival_load=cur_load + self.demands[cus_idx],
            arrival_time=finish_time,
            arrival_soc=arrive_soc2,
            path_nodes=path_nodes2,
        )

    # ------------------------ greedy choice ------------------------

    def find_next_customer(self) -> Optional[Tuple[int, PathInfo]]:
        cur_pos = int(self.state["id"])
        remain = [i for i in range(self.num_customers) if not self.visited[i]]
        if not remain:
            return None

        # greedy ordering: closest by Euclidean distance from current node
        scored = [(self.distance_matrix[cur_pos][1 + i], i) for i in remain]
        scored.sort()

        for _, cus_idx in scored:
            info = self.shortest_time_cur_to_customer(cur_pos, cus_idx)
            if info is None:
                continue
            return cus_idx, info

        return None

    def update_state_after_customer(self, cus_idx: int, info: PathInfo) -> None:
        self.last_move_path = info.path_nodes
        self.state["id"] = 1 + cus_idx
        self.state["time"] = info.arrival_time
        self.state["soc"] = info.arrival_soc
        self.state["load"] = info.arrival_load
        self.visited[cus_idx] = True

    # ------------------------ multi-vehicle solve (parallel vehicles) ------------------------

    def reset_new_vehicle(self) -> None:
        """
        Start a NEW vehicle (parallel) at depot:
          - time resets to working_start
          - full battery
          - empty load
        """
        self.state["id"] = self.depot_node
        self.state["time"] = self.working_start
        self.state["soc"] = self.fuel_cap
        self.state["load"] = 0.0

    def solve(self) -> List[int]:
        """
        Return FULL path as a global-node sequence, e.g. [0, 1, 4, 2, 0, 3, 0].

        IMPORTANT: multi-vehicle semantics (parallel vehicles):
          - Each time we "return to depot and reset", it means we dispatch a new vehicle.
          - Therefore, we reset time to working_start for that new vehicle.
          - Each customer is visited at most once globally.
        """
        full_route: List[int] = []

        def append_move(move: List[int]) -> None:
            if not move:
                return
            if not full_route:
                full_route.extend(move)
            else:
                if full_route[-1] == move[0]:
                    full_route.extend(move[1:])
                else:
                    full_route.extend(move)

        # keep launching new vehicles until all customers visited or remaining infeasible
        while not all(self.visited):
            # launch a new vehicle
            self.reset_new_vehicle()
            append_move([self.depot_node])

            visited_this_vehicle = 0

            while True:
                picked = self.find_next_customer()

                if picked is None:
                    # close this vehicle route by returning to depot
                    cur_id = int(self.state["id"])
                    cur_t = float(self.state["time"])
                    cur_soc = float(self.state["soc"])

                    if cur_id != self.depot_node:
                        ret = self._return_to_depot_path(cur_id, cur_t, cur_soc)
                        if ret is None:
                            # should be rare due to "can return after customer" filter
                            append_move([cur_id, self.depot_node])
                        else:
                            back_path, _arrive_t = ret
                            append_move(back_path)
                    break

                cus_idx, info = picked
                append_move(info.path_nodes)
                self.update_state_after_customer(cus_idx, info)
                visited_this_vehicle += 1

            # if this vehicle couldn't visit anyone, remaining customers are infeasible
            if visited_this_vehicle == 0:
                break

        # ensure final ends at depot (in case last vehicle route ended away from depot)
        if full_route and full_route[-1] != self.depot_node:
            # attempt a safe return using current state
            cur_id = int(self.state["id"])
            cur_t = float(self.state["time"])
            cur_soc = float(self.state["soc"])
            ret = self._return_to_depot_path(cur_id, cur_t, cur_soc)
            if ret is not None:
                back_path, _arrive_t = ret
                append_move(back_path)
            else:
                append_move([full_route[-1], self.depot_node])

        str_route = [self.id_strs[full_route[i]] for i in range(len(full_route))]
        array_route = np.array(str_route)
        route_idx = np.where(array_route == "D0")[0].tolist()

        routes = []
        for i in range(1, len(route_idx)):
            start = route_idx[i-1]
            end = route_idx[i]
            cur_route = str_route[start:end] + ["D0"]
            routes.append("->".join(cur_route))
        self.global_value = sum([self.distance_matrix[full_route[i]][full_route[i - 1]] for i in range(1, len(full_route))])
        
        return routes

