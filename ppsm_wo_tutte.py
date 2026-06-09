import networkx as nx
import bisect
from typing import List, Tuple, Set, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque
from collections import defaultdict
from itertools import product
from copy import deepcopy



def load_graph_from_txt(file_path):
    """
    从txt文件中读取边并构建无向图
    每行格式: u v
    """
    G = nx.Graph()
    with open(file_path, "r") as f:
        for line in f:
            u, v = line.strip().split()
            G.add_edge(int(u), int(v))  # 节点转为整数
    return G


def is_tree(g):
    """
    检查一个图是否为树。
    对于一个连通图，当且仅当边数等于节点数减一时，它是一棵树。
    """
    if not nx.is_connected(g):
        return False
    return len(g.edges) == len(g.nodes) - 1

def custom_graph_partition(graph):
    full_graph = graph.copy()
    final_subgraphs = []
    final_cut_edges = set()

    # 1. 独立处理每个自然连通分量
    for cc_nodes in nx.connected_components(full_graph):
        cc_graph = full_graph.subgraph(cc_nodes).copy()
        
        # 2. 识别核心节点 (属于包含环的分量)
        real_cores = [set(c) for c in nx.k_edge_components(cc_graph, k=2) if len(c) > 1]
        core_nodes_set = set().union(*real_cores) if real_cores else set()
        
        if not core_nodes_set:
            # 纯树分量保持不变
            final_subgraphs.append(cc_graph)
        else:
            cc_cut_edges = []
            all_bridges = list(nx.bridges(cc_graph))
            
            for u, v in all_bridges:
                # 只有当桥的一端连接核心时，才考虑切割
                if u in core_nodes_set or v in core_nodes_set:
                    # 确定哪一端是核心，哪一端是潜在的树
                    core_end, tree_end = (u, v) if u in core_nodes_set else (v, u)
                    
                    # 模拟删除该边，看 tree_end 侧的规模
                    temp_graph = cc_graph.copy()
                    temp_graph.remove_edge(u, v)
                    
                    # 获取 tree_end 所在的连通分量
                    tree_side_nodes = nx.node_connected_component(temp_graph, tree_end)
                    
                    # 关键判定：如果 tree_side 只有 1 个节点，它是“孤立叶子”，不切
                    # 如果 > 1 个节点，它是“树结构”，切断
                    if len(tree_side_nodes) > 1:
                        cc_cut_edges.append((u, v))
                        final_cut_edges.add(tuple(sorted((u, v))))

            # 3. 执行分割
            split_graph = cc_graph.copy()
            split_graph.remove_edges_from(cc_cut_edges)
            for split_cc_nodes in nx.connected_components(split_graph):
                final_subgraphs.append(full_graph.subgraph(split_cc_nodes).copy())

    return final_cut_edges, final_subgraphs

# ================== MultiGraph 转换工具 ==================
def _to_multigraph(G):
    if isinstance(G, nx.MultiGraph):
        MG = nx.MultiGraph()
        MG.add_nodes_from(G.nodes())
        for u, v, k, d in G.edges(keys=True, data=True):
            MG.add_edge(u, v, key=k, **d)
        return MG
    else:
        MG = nx.MultiGraph()
        MG.add_nodes_from(G.nodes())
        for u, v in G.edges():
            MG.add_edge(u, v)
        return MG


# ================== Tutte 矩形近似（仅最大指数） ==================
def approx_tutte_rectangle_maxonly(G):
    """
    返回矩形近似（仅计算 xmax 和 ymax；不包含 xmin/ymin）:
        xmax = |V| - c(G)    (rank)
        ymax = |E| - |V| + c(G) (nullity)
    返回格式: {"xmax": int, "ymax": int}
    """
    MG = _to_multigraph(G)
    n = MG.number_of_nodes()
    m = MG.number_of_edges()

    if n == 0:
        return {"xmax": 0, "ymax": 0}

    num_components = nx.number_connected_components(MG)

    # xmax / ymax 计算（线性时间）
    rank = n - num_components
    nullity = m - n + num_components

    return {"xmax": max(0, rank), "ymax": max(0, nullity)}


def tutte_from_bridge_decomposition(cut_edges, subgraphs):
    subgraph_map = []
    xmax_sum = 0
    ymax_sum = 0

    for i, sg in enumerate(subgraphs):
        rect = compute_rect_subgraph_maxonly(sg)

        subgraph_map.append((i, sg, rect))
        xmax_sum += rect["xmax"]
        ymax_sum += rect["ymax"]

    k = len(cut_edges) if cut_edges else 0
    total_rect = {"xmax": xmax_sum + k, "ymax": ymax_sum}

    return subgraph_map, total_rect

# ================== 并行子图矩形计算 ==================
def compute_rect_subgraph_maxonly(G):
    return approx_tutte_rectangle_maxonly(G)


def tutte_from_bridge_decomposition_parallel(cut_edges, subgraphs, max_workers=None):
    """
    输入:
        cut_edges : 割边集合（仅用于贡献 x^len(cut_edges) 因子到 xmax）
        subgraphs : 子图列表（networkx Graph / MultiGraph 对象）
    输出:
        subgraph_map : [(id, 子图, {"xmax", "ymax"}), ...]
        total_rect   : {"xmax", "ymax"}
    说明:
        - 仅计算并返回 xmax 和 ymax（不再返回 xmin/ymin）。
        - total_rect 中的 xmax 会增加 len(cut_edges)（对应乘以 x^{len(cut_edges)}）。
    """
    subgraph_map = []

    # 并行提交任务
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(compute_rect_subgraph_maxonly, sg): i
            for i, sg in enumerate(subgraphs)
        }

        results = {}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()

    # 聚合最大值
    xmax_sum = 0
    ymax_sum = 0

    for i in range(len(subgraphs)):
        rect = results[i]
        sg = subgraphs[i]

        subgraph_map.append((i, sg, rect))

        xmax_sum += rect["xmax"]
        ymax_sum += rect["ymax"]

    # 割边对 x 的贡献（x^k）
    k = len(cut_edges) if cut_edges is not None else 0

    total_rect = {"xmax": xmax_sum + k, "ymax": ymax_sum}

    return subgraph_map, total_rect

#割块树构建
def build_block_cut_tree(cut_edges, subgraph_map):
    """
    根据割边集合和子图ID映射表构建割块树。
    
    Args:
        cut_edges (set): 割边的集合。
        subgraph_map (list): 子图ID映射表 [(ID, 子图结构, 多项式), ...]。
    
    Returns:
        nx.Graph: 割块树。节点是子图ID，边是割边。
    """
    block_cut_tree = nx.Graph()
    
    # 步骤1: 将每个子图ID作为割块树的节点
    for subgraph_id, _, _ in subgraph_map:
        block_cut_tree.add_node(subgraph_id)
        
    # 步骤2: 为每个割边找到它连接的两个子图，并在割块树中添加一条边
    for u, v in cut_edges:
        subgraph_id1 = -1
        subgraph_id2 = -1
        
        # 查找 u 和 v 所在的子图ID
        for subgraph_id, subgraph_struct, _ in subgraph_map:
            if u in subgraph_struct.nodes:
                subgraph_id1 = subgraph_id
            if v in subgraph_struct.nodes:
                subgraph_id2 = subgraph_id
            
            # 找到两个子图ID后，添加边并跳出循环以提高效率
            if subgraph_id1 != -1 and subgraph_id2 != -1:
                break
        
        # 确保找到两个不同的子图ID
        if subgraph_id1 != -1 and subgraph_id2 != -1 and subgraph_id1 != subgraph_id2:
            # 可以在边的属性中存储信息，例如：weight=1
            block_cut_tree.add_edge(subgraph_id1, subgraph_id2)
            
    return block_cut_tree

#索引构建
def build_rect_dominance_index_keep_self(subgraph_map):
    """
    构建基于 rect 的支配索引（支配集合中保留自身）。
    输入:
      subgraph_map: iterable of (id, subgraph, rect) OR (id, rect, subgraph)
                    rect 至少包含 'xmax', 'ymax'
    输出:
      sorted_rects_with_dominators:
          list of ((xmax, ymax), set(dominator_ids))
          rect 按 (xmax,ymax) 升序排列
      rect_to_doms:
          dict: (xmax,ymax) -> set(dominator_ids)
    """
    items = []
    for entry in subgraph_map:
        if len(entry) != 3:
            raise ValueError("each entry must be (id, subgraph, rect) or (id, rect, subgraph)")
        a, b, c = entry
        if isinstance(b, dict) and 'xmax' in b:
            sid, rect = a, b
        elif isinstance(c, dict) and 'xmax' in c:
            sid, rect = a, c
        else:
            raise ValueError("cannot find rect dict with xmax/ymax")
        items.append((sid, int(rect["xmax"]), int(rect["ymax"])))

    if not items:
        return [], {}

    # 按强度排序（大 -> 小）
    items.sort(key=lambda t: (t[1], t[2]), reverse=True)

    rect_of_id = {sid: (x, y) for sid, x, y in items}
    dominators_by_id = {sid: set() for sid, _, _ in items}

    seen_by_y = defaultdict(list)
    seen_y_keys = []

    for sid, xmax, ymax in items:
        # 先加入 seen（保证自身也视为支配）
        if ymax not in seen_by_y:
            bisect.insort(seen_y_keys, ymax)
        seen_by_y[ymax].append(sid)

        # 找 y >= 当前 ymax 的所有 seen
        idx = bisect.bisect_left(seen_y_keys, ymax)
        doms = set()
        for key in seen_y_keys[idx:]:
            doms.update(seen_by_y[key])

        # 与之前不同：不再删除自身
        dominators_by_id[sid] = doms

    # 输出（按 rect 升序）
    id_rect_list = [(sid, rect_of_id[sid]) for sid, _, _ in items]
    id_rect_list.sort(key=lambda t: (t[1][0], t[1][1]))

    sorted_rects_with_dominators = []
    rect_to_doms = {}

    for sid, rect in id_rect_list:
        rect_tuple = (rect[0], rect[1])
        doms = dominators_by_id[sid]

        sorted_rects_with_dominators.append((rect_tuple, set(doms)))

        if rect_tuple not in rect_to_doms:
            rect_to_doms[rect_tuple] = set(doms)
        else:
            rect_to_doms[rect_tuple].update(doms)

    return sorted_rects_with_dominators

#基于支配集索引的快速查找

def build_aux_from_sorted_index(sorted_rects_with_dominators: List[Tuple[Tuple[int,int], Set[int]]]):
    """
    为已排序的索引构建辅助数组 xs 和 suffix_max_y。
    返回 (xs, ys, suffix_max_y)
    - xs: list of xmax
    - ys: list of ymax (aligned)
    - suffix_max_y[i] = max(ys[i:])  (用于快速判定是否存在 y >= qy 在区间 [i, end])
    """
    xs = [rect[0] for rect, _ in sorted_rects_with_dominators]
    ys = [rect[1] for rect, _ in sorted_rects_with_dominators]
    n = len(ys)
    suffix_max_y = [0] * n
    cur = -10**30
    for i in range(n-1, -1, -1):
        if ys[i] > cur:
            cur = ys[i]
        suffix_max_y[i] = cur
    return xs, ys, suffix_max_y

def find_passing_ids_from_index(query_rect: Dict[str,int],
                                sorted_rects_with_dominators: List[Tuple[Tuple[int,int], Set[int]]]):
    """
    输入:
      - query_rect: {"xmax": int, "ymax": int}
      - sorted_rects_with_dominators: [ ((xmax,ymax), set(dominator_ids)), ... ]
         必须按 (xmax,ymax) 升序排列
    输出:
      - passing_ids: set of ids whose rect >= query_rect (i.e., candidates that may contain the query)
    说明:
      - 该函数对 index 做 O(log n) 的二分 + O(k) 的扫描（k 为从第一个 x>=qx 到末尾需要检查的元素数目）。
      - 若需要大量重复查询，建议把 build_aux_from_sorted_index 的结果缓存起来，以避免重复构造 suffix_max_y。
    """
    if not sorted_rects_with_dominators:
        return set()

    qx = int(query_rect["xmax"])
    qy = int(query_rect["ymax"])

    # 构造辅助数组（可改为在外部预建并传入以避免重复）
    xs, ys, suffix_max_y = build_aux_from_sorted_index(sorted_rects_with_dominators)

    # 找到第一个 x >= qx 的索引 i
    i = bisect.bisect_left(xs, qx)
    if i >= len(xs):
        return set()  # 所有 rect 的 xmax 都 < qx

    # 若后缀最大 y 都 < qy，则不存在满足 y>=qy 的 rect
    if suffix_max_y[i] < qy:
        return set()

    # 否则，从 i 向右扫描，当 rect.y >= qy 时并入其 dominator id 集合
    passing_ids = set()
    n = len(xs)
    for j in range(i, n):
        ry = ys[j]
        if ry >= qy:
            doms = sorted_rects_with_dominators[j][1]
            # doms 是一个 set，包含支配该 rect 的所有 id（含自身）
            passing_ids.update(doms)
        # 小剪枝：如果 suffix_max_y[j] < qy，则后面都不会满足，提前终止
        if j + 1 < n and suffix_max_y[j+1] < qy:
            break

    return passing_ids

#判定是否存在可能的解
def maybe_contains_by_rect(query_rect, data_rect):
    """
    基于矩形近似判定：数据图是否“可能包含”查询图的同胚子图。
    规则（必要条件，不充分）：
      - 若 query.xmax <= data.xmax 且 query.ymax <= data.ymax 则返回 (True, reasons)
        表示数据图可能包含查询图（需进一步精确验证以确认）。
      - 否则返回 (False, reasons) 表示可以安全地判定“数据图不包含查询图”。

    输入：
      - query_rect: dict with keys "xmax", "ymax" (ints)
      - data_rect:  dict with keys "xmax", "ymax" (ints)

    返回：
      - (possible: bool, reasons: list[str])
    """
    reasons = []

    qx = query_rect.get("xmax")
    qy = query_rect.get("ymax")
    dx = data_rect.get("xmax")
    dy = data_rect.get("ymax")

    # 输入校验（简单）
    if qx is None or qy is None or dx is None or dy is None:
        return False

    if qx > dx:
        reasons.append(f"query.xmax={qx} > data.xmax={dx}  -> 必然不包含")
    if qy > dy:
        reasons.append(f"query.ymax={qy} > data.ymax={dy}  -> 必然不包含")

    if reasons:
        return False

    # 两者均不超过数据上界：只能判定为可能包含（必要条件满足）
    reasons.append(f"query.xmax={qx} <= data.xmax={dx} 且 query.ymax={qy} <= data.ymax={dy} -> 可能包含")
    return True


#查询处理（树结构查询图）
def get_tutte_poly_from_map(subgraph_map, subgraph_id):
    """
    根据ID从子图映射表中获取Tutte多项式。
    """
    for sg_id, _, poly in subgraph_map:
        if sg_id == subgraph_id:
            return poly
    return None

#从索引中返回候选解
def find_candidates_from_index_optimized(query_poly, tutte_index, subgraph_map):
    """
    根据查询图的 Tutte 多项式和 Tutte 多项式索引，查找满足条件的候选子图集合。
    
    Args:
        query_poly (dict): 查询图的 Tutte 多项式。
        tutte_index (dict): Tutte多项式一维索引。
        subgraph_map (list): 子图ID映射表。
    
    Returns:
        set: 满足条件的候选子图ID集合。
    """
    if not query_poly:
        return set()

    # 步骤1: 使用索引进行初步筛选，找到所有可能匹配的子图ID
    possible_candidates = set()
    for (q_x_exp, q_y_exp), q_coeff in query_poly.items():
        for (idx_x_exp, idx_y_exp), id_weight_set in tutte_index.items():
            if idx_x_exp >= q_x_exp and idx_y_exp >= q_y_exp:
                for sg_id, coeff in id_weight_set:
                    if coeff >= q_coeff:
                        possible_candidates.add(sg_id)

    # 步骤2: 对每个初步候选子图进行精确的、不重复的项匹配判定
    final_candidates = set()
    for sg_id in possible_candidates:
        candidate_poly = get_tutte_poly_from_map(subgraph_map, sg_id)
        if candidate_poly and check_subgraph_match(query_poly, candidate_poly):
            final_candidates.add(sg_id)

    return final_candidates

def check_subgraph_match(query_poly, candidate_poly):
    """
    检查一个候选多项式是否满足查询多项式的所有项，且匹配项不重复。
    这个函数实现了你提出的核心判定逻辑。
    """
    if not query_poly:
        return True
    
    query_terms = sorted(query_poly.items(), key=lambda x: (x[0][0], x[0][1], x[1]), reverse=True)
    candidate_terms = sorted(candidate_poly.items(), key=lambda x: (x[0][0], x[0][1], x[1]), reverse=True)
    
    matched_candidate_indices = set()
    
    for q_idx, ((q_x, q_y), q_c) in enumerate(query_terms):
        found_match = False
        for c_idx, ((c_x, c_y), c_c) in enumerate(candidate_terms):
            if c_idx not in matched_candidate_indices:
                if c_x >= q_x and c_y >= q_y and c_c >= q_c:
                    matched_candidate_indices.add(c_idx)
                    found_match = True
                    break
        if not found_match:
            return False
            
    return True

def build_tutte_poly_x_exp_from_nodes(nodes, subgraph_map):
    total_x_exp = 0
    for sg_id in nodes:
        # 获取该子图的 rect 而非 poly
        for sid, sg, rect in subgraph_map:
            if sid == sg_id:
                total_x_exp += int(rect["xmax"])
                break
    return total_x_exp

def get_graph_depth(G):
    """
    计算图的深度，对于无环图（树）为最长路径长度。
    """
    if not G.nodes():
        return 0
    
    # 修复: nx.shortest_path_length 返回一个生成器
    # 必须迭代它来获取路径信息
    max_depth = 0
    # iter_nodes 是一个包含 (source, paths_dict) 的生成器
    iter_nodes = nx.shortest_path_length(G)
    for source, paths in iter_nodes:
        for target, path_length in paths.items():
            max_depth = max(max_depth, path_length)
    return max_depth

def get_max_degree_in_subgraph(subgraph_id, subgraph_map, data_bridges):
    """
    根据子图ID映射表和割边集合，获取子图的最大度。
    """
    sg = next((s for sid, s, _ in subgraph_map if sid == subgraph_id), None)
    if not sg:
        return 0
    
    max_degree = 0
    subgraph_nodes = set(sg.nodes())
    
    for node in subgraph_nodes:
        current_degree = sg.degree(node)
        
        for u, v in data_bridges:
            if u == node or v == node:
                current_degree += 1
        max_degree = max(max_degree, current_degree)
        
    return max_degree

def handle_tree_query(query_graph, subgraph_map, data_block_cut_tree, tutte_index, k, data_bridges):
    final_candidates = set()
    is_processed = [False] * len(subgraph_map)

    # 这里使用 approx_tutte_rectangle_maxonly 返回的是 rect 而非多项式
    query_rect = approx_tutte_rectangle_maxonly(query_graph)   # -> {"xmax": int, "ymax": int}
    # 保证类型为 int
    qx = int(query_rect.get("xmax", 0))
    qy = int(query_rect.get("ymax", 0))

    # 查询图最大度与深度（保持不变）
    query_max_degree = max(dict(query_graph.degree()).values()) if query_graph.nodes else 0
    query_graph_depth = get_graph_depth(query_graph)

    # query_max_x_exp 对应 query_rect 的 xmax
    query_max_x_exp = qx

    # 1. 初始筛选：用 rect 索引查出可能通过的子图 id
    # 注意：find_passing_ids_from_index 接受 query_rect（含 xmax,ymax）
    initial_candidates = find_passing_ids_from_index({"xmax": qx, "ymax": qy}, tutte_index)

    for sg_id in initial_candidates:
        if 0 <= sg_id < len(is_processed):
            is_processed[sg_id] = True
    final_candidates.update(initial_candidates)

    print(f"步骤1: 初始筛选找到的候选子图: {initial_candidates}")

    # 2. 在割块树上以 k 距离扩展候选（不变）
    for sg_id in list(initial_candidates):
        q = deque([(sg_id, 0)])
        visited_in_bfs = {sg_id}
        while q:
            curr_sg_id, distance = q.popleft()
            if distance >= k:
                continue

            for neighbor_id in data_block_cut_tree.neighbors(curr_sg_id):
                if neighbor_id not in visited_in_bfs:
                    visited_in_bfs.add(neighbor_id)
                    if 0 <= neighbor_id < len(is_processed):
                        is_processed[neighbor_id] = True
                        final_candidates.add(neighbor_id)
                        q.append((neighbor_id, distance + 1))

    print(f"步骤2: 基于距离 k 扩展后的候选子图: {final_candidates}")

    # 3. 尝试合并其他未处理的子图（保持原逻辑，仅用 query_max_x_exp 来比较）
    for sg_id in range(len(subgraph_map)):
        if is_processed[sg_id]:
            continue

        max_degree = get_max_degree_in_subgraph(sg_id, subgraph_map, data_bridges)
        if max_degree < query_max_degree:
            # 度不足，跳过
            is_processed[sg_id] = True
            continue

        # 从当前根节点开始，在割块树上扩展指定深度，得到临时候选子图集
        temp_candidate_set = {sg_id}
        q = deque([(sg_id, 0)])
        visited_in_extension = {sg_id}

        while q:
            curr_sg_id, distance = q.popleft()
            if distance >= query_graph_depth + k:
                continue
            for neighbor_id in data_block_cut_tree.neighbors(curr_sg_id):
                if neighbor_id not in visited_in_extension:
                    visited_in_extension.add(neighbor_id)
                    temp_candidate_set.add(neighbor_id)
                    q.append((neighbor_id, distance + 1))

        # 单次验证：检查临时子图集是否满足查询图的 x_max 指数之和要求
        merged_poly_x_exp = build_tutte_poly_x_exp_from_nodes(temp_candidate_set, subgraph_map)

        # merged_poly_x_exp 是 int， query_max_x_exp 也保证为 int，上面已修正
        if merged_poly_x_exp >= query_max_x_exp:
            print(f"步骤3: 发现新的组合解：{temp_candidate_set}")
            final_candidates.update(temp_candidate_set)
            for node_id in temp_candidate_set:
                if 0 <= node_id < len(is_processed):
                    is_processed[node_id] = True

        # 无论是否找到解，都将根节点标记为已处理
        is_processed[sg_id] = True

    return final_candidates

#查询处理（包含割边的非纯树查询图）
def get_path_info_cached(data_bct, d_u, d_v, path_nodes_filter, cache):
    """
    带缓存的路径计算函数。
    """
    if d_u == d_v:
        return None, None

    # 规范化缓存键，因为图是无向的
    cache_key = tuple(sorted((d_u, d_v)))
    if cache_key in cache:
        path_length, internal_nodes = cache[cache_key]
    else:
        try:
            path = nx.shortest_path(data_bct, source=d_u, target=d_v)
            path_length = len(path) - 1
            internal_nodes = set(path[1:-1])
            cache[cache_key] = (path_length, internal_nodes)
        except nx.NetworkXNoPath:
            cache[cache_key] = (None, None) # 缓存不存在的路径
            return None, None
            
    # 如果路径有效，才进行不相交性检查
    if path_length is None:
        return None, None
        
    if internal_nodes.intersection(path_nodes_filter):
        return None, None

    return path_length, internal_nodes



# --- 核心枚举函数 (遵循你的 BFS 驱动策略) ，返回可行的候选解组合---

def _backtracking_search_ordered(
    node_index,
    traversal_order,
    bfs_parents,
    match_map,
    path_nodes_filter,
    current_path_len,
    max_path_len,
    data_bct,
    candidates,
    solutions,
    path_cache
):
    """
    按照预先计算的BFS遍历顺序进行回溯搜索。
    """
    # 终止条件：所有节点都已成功匹配
    if node_index == len(traversal_order):
        solutions.append((match_map.copy(), current_path_len))
        return

    q_curr = traversal_order[node_index]
    q_parent = bfs_parents.get(q_curr) # 获取当前节点在BFS树中的父节点

    # 遍历当前查询节点的所有候选数据节点
    for d_curr in candidates.get(q_curr, set()):
        # 1. 注入性检查: 候选的数据节点不能已经被使用
        if d_curr in match_map.values():
            continue

        path_length_to_parent = 0
        new_path_nodes = set()

        # 2. 结构/路径约束检查 (仅对非根节点)
        if q_parent is not None:
            d_parent = match_map[q_parent]
            
            path_length, path_nodes_to_add = get_path_info_cached(
                data_bct, d_parent, d_curr, path_nodes_filter, path_cache
            )

            if path_length is None:
                continue
            
            path_length_to_parent = path_length
            new_path_nodes = path_nodes_to_add

        # 3. 总路径长度约束检查
        if (current_path_len + path_length_to_parent) > max_path_len:
            continue

        # --- 所有检查通过，进行扩展 ---
        match_map[q_curr] = d_curr
        
        # 递归调用，处理遍历顺序中的下一个节点
        _backtracking_search_ordered(
            node_index + 1,
            traversal_order,
            bfs_parents,
            match_map,
            path_nodes_filter.union(new_path_nodes),
            current_path_len + path_length_to_parent,
            max_path_len,
            data_bct,
            candidates,
            solutions,
            path_cache
        )
        
        # --- 回溯 ---
        del match_map[q_curr]


def find_valid_combos_optimized(query_bct, data_bct, candidates, k):
    """
    优化的结构匹配函数（已修正遍历逻辑）。
    返回: List[Tuple[Dict, int]]，每个元素是 (匹配字典, 使用的总路径长度)。
    """
    if not query_bct.nodes():
        return []

    # 1. 根节点选择
    try:
        q_root = min(query_bct.nodes(), key=lambda q_node: len(candidates.get(q_node, [])))
    except (ValueError, TypeError): # 添加TypeError以防candidates为空
        return []

    # 2. 生成固定的BFS遍历顺序和父子关系
    traversal_order = list(nx.bfs_tree(query_bct, source=q_root).nodes())
    bfs_parents = {child: parent for parent, child in nx.bfs_edges(query_bct, source=q_root)}

    max_allowed_path_length = query_bct.number_of_edges() + k
    solutions = []
    path_cache = {}

    # 3. 启动回溯搜索
    # 只需要为根节点启动循环，后续节点由递归函数按顺序处理
    _backtracking_search_ordered(
        node_index=0,  # 从遍历顺序的第一个节点（即根节点）开始
        traversal_order=traversal_order,
        bfs_parents=bfs_parents,
        match_map={},   # 初始匹配为空
        path_nodes_filter=set(),
        current_path_len=0,
        max_path_len=max_allowed_path_length,
        data_bct=data_bct,
        candidates=candidates,
        solutions=solutions,
        path_cache=path_cache
    )
        
    return solutions
         
def extend_and_validate_tree_node(
    q_tree_node_id,           # 当前处理的查询图树节点ID
    q_tree_graph,             # 查询图树节点的实际图对象
    q_original_bct,           # 原始查询图割块树 (包含树节点)
    data_bct,                 # 数据图割块树
    main_combo_match_map,     # 当前的主干匹配：{Q_non_tree_id: D_node_id}
    data_subgraph_map,        # 数据图子图映射
    main_combo_edges_used,    # 主干匹配已消耗的路径长度 |E_non_tree_bct_paths|
    total_allowed_k,          # 原始查询图允许的总路径长度 |E_Q| + k
    data_bridges              # 数据图的割边集合
):
    """
    尝试在数据图割块树上扩展以匹配单个查询图树节点，并增加了最大度约束检查。

    返回: (是否找到有效扩展, 匹配的数据子图ID集合)
    """
    
    # 1. 确定扩展起点、约束和查询图需求
    q_neighbor_id = next(iter(q_original_bct.neighbors(q_tree_node_id)), None)
    if q_neighbor_id is None or q_neighbor_id not in main_combo_match_map:
        return False, set()

    d_root_id = main_combo_match_map[q_neighbor_id]
    
    q_tree_edges = q_tree_graph.number_of_edges()
    q_tree_max_degree = max(dict(q_tree_graph.degree()).values(), default=0) 
    q_tree_depth = get_graph_depth(q_tree_graph) 
    
    extension_limit = (total_allowed_k - main_combo_edges_used) + q_tree_depth

    # 2. 扩展搜索 (BFS)
    q_bfs = deque([(d_root_id, 0, {d_root_id})]) 
    best_match = None
    main_combo_data_nodes = set(main_combo_match_map.values())

    while q_bfs:
        curr_d_id, distance, temp_subgraph_set = q_bfs.popleft()
        
        if distance > extension_limit:
            continue
        
        # --- 候选解判定开始 ---
        
        # a. 检查 x 最高次幂约束 (边数检查)
        x_exp_temp = build_tutte_poly_x_exp_from_nodes(temp_subgraph_set, data_subgraph_map)
        
        if x_exp_temp >= q_tree_edges:
            
            # b. **最大度约束检查** (调用外部函数)
            d_max_degree = 0
            for d_sg_id in temp_subgraph_set:
                d_max_degree = max(d_max_degree, 
                                            get_max_degree_in_subgraph(d_sg_id, data_subgraph_map, data_bridges))
            
            if d_max_degree >= q_tree_max_degree:
                
                # c. 割块树连接检查 (确保 d_root_id 仍是连接点，此处简化)
                
                # 如果所有验证通过，则找到最佳匹配
                best_match = temp_subgraph_set
                break
            
        # 广度优先搜索
        for neighbor_d_id in data_bct.neighbors(curr_d_id):
            # 扩展方向必须是过滤集合之外的子图节点 and 不属于主干匹配的子图节点
            if neighbor_d_id not in temp_subgraph_set and neighbor_d_id not in main_combo_data_nodes:
                new_set = temp_subgraph_set.union({neighbor_d_id})
                q_bfs.append((neighbor_d_id, distance + 1, new_set))
    
    # 3. 返回结果
    if best_match:
        return True, best_match
    else:
        return False, set()


# --- 主函数 ---

def handle_query_with_tree_nodes(
    q_original_bct, data_bct, subgraph_map_q, data_subgraph_map, k, 
    q_subgraph_candidates_map, q_is_tree_map, data_bridges
):
    final_candidates = set()
    
    # 步骤 1: 分离 (代码不变)
    q_non_tree_nodes = {q_id for q_id, is_tree in q_is_tree_map.items() if not is_tree}
    q_tree_nodes = set(q_is_tree_map.keys()) - q_non_tree_nodes
    q_non_tree_candidates = {q_id: q_subgraph_candidates_map.get(q_id, set()) for q_id in q_non_tree_nodes}
    
    q_non_tree_bct = q_original_bct.copy()
    q_non_tree_bct.remove_nodes_from(q_tree_nodes)
    
    # 步骤 2: 使用优化后的函数进行主干匹配
    print("\n--- 结构匹配: 非树主干部分 (优化版) ---")
    # non_tree_solutions 是一个 (match_map, path_len) 的列表
    non_tree_solutions = find_valid_combos_optimized(
        q_non_tree_bct, 
        data_bct, 
        q_non_tree_candidates, 
        k
    )
    
    if not q_tree_nodes:
        # 纯非树查询
        for match_map, _ in non_tree_solutions:
            final_candidates.update(match_map.values())
        return final_candidates
        
    # 步骤 3: 树节点扩展
    print(f"--- 扩展: 针对 {len(non_tree_solutions)} 组主干匹配进行树节点扩展 ---")
    
    total_allowed_k = q_original_bct.number_of_edges() + k
    
    # *** 关键修改: 直接从 non_tree_solutions 获取路径长度 ***
    for non_tree_match_map, main_combo_edges_used in non_tree_solutions:
        
        temp_combo_data_set = set(non_tree_match_map.values()) 
        combo_is_valid = True
        
        for q_tree_id in q_tree_nodes:
            q_tree_graph = subgraph_map_q.get(q_tree_id, {}).get('graph')
            if not q_tree_graph: continue

            success, tree_match_data_set = extend_and_validate_tree_node(
                q_tree_id, q_tree_graph, q_original_bct, data_bct, 
                non_tree_match_map, data_subgraph_map,
                main_combo_edges_used, # 直接使用
                total_allowed_k,
                data_bridges
            )

            if success:
                temp_combo_data_set.update(tree_match_data_set)
            else:
                combo_is_valid = False
                break
        
        if combo_is_valid:
            final_candidates.update(temp_combo_data_set)
            
    return final_candidates

def process_non_tree_query(query_graph, data_subgraph_map, data_block_cut_tree, tutte_index, k, data_bridges):
    """
    处理包含割边的非纯树查询图。
    
    整合了初始筛选、快速剪枝、主干结构匹配和树节点动态扩展的完整流程。
    
    输出: 最终候选解集合 (数据图子图ID的并集)。
    """
    print("数据图割块树：",list(data_block_cut_tree.nodes), list(data_block_cut_tree.edges))
    # ----------------------------------------------------
    # 步骤 1: 查询图预处理与初始筛选 (代码不变)
    # ----------------------------------------------------
    print("--- 步骤 1: 查询图分解与初始筛选 ---")
    
    cut_edges_q, subgraphs_q = custom_graph_partition(query_graph)
    subgraph_map_q_list, graph_poly_q = tutte_from_bridge_decomposition_parallel(cut_edges_q, subgraphs_q)
    
    subgraph_map_q = {}
    for sg_id, sg_graph, sg_poly in subgraph_map_q_list:
        subgraph_map_q[sg_id] = {'graph': sg_graph, 'poly': sg_poly}
    
    q_original_bct = build_block_cut_tree(cut_edges_q, subgraph_map_q_list) 
    q_subgraph_candidates_map = {}
    q_is_tree_map = {}
    print("查询图割块树：",list(q_original_bct.nodes), list(q_original_bct.edges))

    for sg_id_q, data in subgraph_map_q.items():
        is_tree_flag = is_tree(data['graph']) 
        q_is_tree_map[sg_id_q] = is_tree_flag
        q_subgraph_candidates_map[sg_id_q] = find_passing_ids_from_index(data['poly'], tutte_index)

    print("初始候选解：",q_subgraph_candidates_map)
    q_non_tree_nodes = {q_id for q_id, is_tree_flag in q_is_tree_map.items() if not is_tree_flag}
    q_tree_nodes = {q_id for q_id, is_tree_flag in q_is_tree_map.items() if is_tree_flag}
    
    final_candidates = set()

    # ----------------------------------------------------
    # 步骤 2: 初始联合验证与无解剪枝
    # ----------------------------------------------------
    print("--- 步骤 2: 初始联合验证与无解剪枝 ---")
    
    for q_id in q_non_tree_nodes:
        if not q_subgraph_candidates_map.get(q_id):
            print(f"剪枝: 非树节点 {q_id} 没有初始候选解。返回无解。")
            return set() 
            
    has_unmatched_tree_node = any(not q_subgraph_candidates_map.get(q_id) for q_id in q_tree_nodes)
            
    if not has_unmatched_tree_node:
        print("快速验证: 所有节点均有初始候选解，尝试完整结构匹配...")
        
        all_node_solutions = find_valid_combos_optimized(
            q_original_bct, 
            data_block_cut_tree, 
            q_subgraph_candidates_map, 
            k
        )
        print(list(all_node_solutions))
        if all_node_solutions:
            print(f"快速验证成功: 找到 {len(all_node_solutions)} 组完整匹配。")
            
            # *** 核心修正 1: 解包元组 ***
            # 原代码: for match_map in all_node_solutions:
            for match_map, _ in all_node_solutions: # 使用 (match_map, _) 来解包
                final_candidates.update(match_map.values())
        else:
             print("快速验证失败: 无完全基于初始候选解的匹配。")

    # ----------------------------------------------------
    # 步骤 3: 主干匹配与树节点扩展
    # ----------------------------------------------------
    
    if not q_tree_nodes:
        print("--- 步骤 3: 纯非树查询处理 ---")
        
        if not final_candidates:
            q_non_tree_bct = q_original_bct.copy()
            q_non_tree_bct.remove_nodes_from(q_tree_nodes)
            q_non_tree_candidates = {q_id: q_subgraph_candidates_map[q_id] for q_id in q_non_tree_nodes}
            
            non_tree_solutions = find_valid_combos_optimized(
                q_non_tree_bct, data_block_cut_tree, q_non_tree_candidates, k
            )

            # *** 核心修正 2: 解包元组 ***
            # 原代码: for match_map in non_tree_solutions:
            for match_map, _ in non_tree_solutions: # 使用 (match_map, _) 来解包
                final_candidates.update(match_map.values())
                
        return final_candidates

    else:
        print("--- 步骤 3: 主干匹配与树节点扩展 ---")
        
        # handle_query_with_tree_nodes 内部已经正确处理了元组，无需修改
        extended_candidates = handle_query_with_tree_nodes(
            q_original_bct, 
            data_block_cut_tree, 
            subgraph_map_q, 
            data_subgraph_map, 
            k, 
            q_subgraph_candidates_map, 
            q_is_tree_map,
            data_bridges
        )
        
        final_candidates.update(extended_candidates)
        return final_candidates
    
#查询处理（总入口）
def dispatch_query_processing(
    query_graph,
    data_subgraph_map,
    data_block_cut_tree,
    tutte_index,
    k,
    data_bridges
):
    """
    根据查询图的类型调度到相应的处理函数。

    Args:
        query_graph (nx.Graph): 查询图。
        data_subgraph_map (dict): 数据图的子图ID映射表。
        data_block_cut_tree (nx.Graph): 数据图的割块树。
        tutte_index (dict): Tutte多项式索引。
        k (int): 路径长度容差。
        data_bridges (set): 数据图的割边集合。

    Returns:
        set: 候选解集合（数据子图ID）。
    """
    print("\n--- Starting New Query ---")
    print(f"Query Graph: {query_graph.number_of_nodes()} nodes, {query_graph.number_of_edges()} edges.")

    # 1. 判断查询图类型
    if not nx.is_connected(query_graph):
        print("Error: Query graph is not connected.")
        return set()

    num_nodes = query_graph.number_of_nodes()
    num_edges = query_graph.number_of_edges()
    
    # 使用nx.bridges的高效实现来查找割边
    bridges = list(nx.bridges(query_graph))

    # 2. 根据类型进行调度
    
    # 情况一: 查询图为单纯的高连通分量 (没有割边)
    if len(bridges) == 0:
        print("Query Type: Pure High-Connectivity Component (No Bridges)")
        print("-> Executing: Direct index lookup")
        
        # a. 计算查询图的Tutte多项式
        query_poly = approx_tutte_rectangle_maxonly(query_graph)
        print(query_poly)
        
        # b. 直接从索引中搜索候选解
        # 注意：函数签名与您提供的不完全一致，这里做了适配
        return find_passing_ids_from_index(query_poly, tutte_index)

    # 情况二: 查询图为纯树 (边数 = 节点数 - 1)
    elif num_edges == num_nodes - 1:
        # 这个条件是判断一个连通图是否为树的充要条件
        print("Query Type: Pure Tree")
        return handle_tree_query(
            query_graph, data_subgraph_map, data_block_cut_tree, tutte_index, k, data_bridges
        )
        
    # 情况三: 包含割边的非纯树 (其他所有情况)
    else:
        print("Query Type: Mixed (High-Connectivity Components connected by Bridges)")
        # process_non_tree_query 是您已经完善的函数
        # 注意：为了让代码能跑通，我将使用您之前代码中的函数名
        # 您提供的函数名是 process_non_tree_query
        return process_non_tree_query(
            query_graph, data_subgraph_map, data_block_cut_tree, tutte_index, k, data_bridges
        )

#构建候选解空间
# def construct_candidate_solution_space(
#     query_graph: nx.Graph,
#     k: int,
#     candidate_subgraph_ids: set,
#     data_subgraph_map: list,
#     data_bridges: set
# ) -> tuple[dict, dict, dict]:
#     """
#     重构后的候选解空间构建函数：预存储所有符合预算的路径。

#     Returns:
#         tuple[dict, dict, dict]: 
#         - vertex_map: {q_node: {d_node1, ...}}
#         - edge_map: {(q_u, q_v): {(d_u, d_v, length), ...}}
#         - path_dict: {(d_u, d_v): [([path_nodes], length), ...]}  <-- 新增：存储具体路径
#     """
    
#     # 1. 恢复候选数据图 (CDG)
#     cdg = nx.Graph() 
#     data_subgraph_dict = {sg_id: sg_graph for sg_id, sg_graph, _ in data_subgraph_map}

#     print("--- 步骤 1: 构建候选数据图 ---")
#     for sg_id in candidate_subgraph_ids:
#         if sg_id in data_subgraph_dict:
#             subgraph = data_subgraph_dict[sg_id]
#             cdg.add_edges_from(subgraph.edges())
#             cdg.add_nodes_from(subgraph.nodes())

#     cdg_nodes = set(cdg.nodes())
#     for u, v in data_bridges:
#         if u in cdg_nodes and v in cdg_nodes:
#             cdg.add_edge(u, v)

#     if not cdg.nodes: return {}, {}, {}

#     # 2. 顶点映射 (保持度约束)
#     query_degree_map = dict(query_graph.degree())
#     candidate_vertex_map = defaultdict(set)
#     print("--- 步骤 2: 构建顶点映射 ---")
#     for q_vid, q_deg in query_degree_map.items():
#         for d_vid in cdg.nodes():
#             if cdg.degree(d_vid) >= q_deg:
#                 candidate_vertex_map[q_vid].add(d_vid)
#         if not candidate_vertex_map[q_vid]: return {}, {}, {}

#     # 3. 构建路径字典与边映射
#     candidate_edge_map = defaultdict(set)
#     candidate_path_dict = {}  # 键: (d_ui, d_uj), 值: list of (path, length)
    
#     # 设定路径最大长度：1 条边对应长度 1，允许额外 k 步
#     max_path_len = 1 + k 

#     print(f"--- 步骤 3: 预提取所有长度 <= {max_path_len} 的候选路径 ---")

#     for q_vi, q_vj in query_graph.edges():
#         query_edge_key = tuple(sorted((q_vi, q_vj)))
#         c_vi = candidate_vertex_map.get(q_vi, set())
#         c_vj = candidate_vertex_map.get(q_vj, set())
        
#         for d_ui, d_uj in product(c_vi, c_vj):
#             if d_ui == d_uj: continue
            
#             # 为了节省空间和计算，对数据节点对进行排序
#             data_pair = tuple(sorted((d_ui, d_uj)))
            
#             # 如果这一对点的路径还没计算过
#             if data_pair not in candidate_path_dict:
#                 valid_paths = []
#                 try:
#                     # 获取所有满足 cutoff 长度的路径
#                     # 注意：cutoff 在 nx 中指节点数，长度为 L 的路径节点数为 L+1
#                     # 但在 cutoff 参数里，有的版本指边数，这里建议使用 all_simple_paths 并手动判断
#                     for path in nx.all_simple_paths(cdg, source=data_pair[0], target=data_pair[1], cutoff=max_path_len):
#                         path_len = len(path) - 1
#                         valid_paths.append((path, path_len))
                    
#                     candidate_path_dict[data_pair] = valid_paths
#                 except nx.NetworkXNoPath:
#                     candidate_path_dict[data_pair] = []

#             # 如果这对点之间存在合法路径，更新 edge_map
#             if candidate_path_dict[data_pair]:
#                 # 记录该查询边可能对应的数据点对及对应的最短路径长度（用于初步剪枝）
#                 min_len = min(p[1] for p in candidate_path_dict[data_pair])
#                 candidate_edge_map[query_edge_key].add((data_pair[0], data_pair[1], min_len))

    
#     for q_edge in query_graph.edges():
#         q_key = tuple(sorted(q_edge))
#         if not candidate_edge_map[q_key]:
#             print(f"剪枝: 查询边 {q_edge} 没有任何合法候选路径。")
#             return {}, {}, {}

#     return dict(candidate_vertex_map), dict(candidate_edge_map), candidate_path_dict

import networkx as nx
from collections import defaultdict, deque
from itertools import product

def construct_candidate_solution_space(
    query_graph: nx.Graph,
    k: int,
    candidate_subgraph_ids: set,
    data_subgraph_map: list,
    data_bridges: set
) -> tuple[dict, dict, dict]:
    """
    基于邻居安全 (Neighbor-Safety) 思想的候选空间构建。
    """
    # 1. 恢复候选数据图 (CDG)
    cdg = nx.Graph() 
    data_subgraph_dict = {sg_id: sg_graph for sg_id, sg_graph, _ in data_subgraph_map}
    for sg_id in candidate_subgraph_ids:
        if sg_id in data_subgraph_dict:
            sg = data_subgraph_dict[sg_id]
            cdg.add_edges_from(sg.edges())
            cdg.add_nodes_from(sg.nodes())
    cdg_nodes = set(cdg.nodes())
    for u, v in data_bridges:
        if u in cdg_nodes and v in cdg_nodes:
            cdg.add_edge(u, v)
    if not cdg.nodes: return {}, {}, {}

    # 2. 初始过滤：度约束
    q_degrees = dict(query_graph.degree())
    cv_map = {q: {d for d in cdg.nodes() if cdg.degree(d) >= deg} for q, deg in q_degrees.items()}

    # 3. 核心改进：邻居安全迭代 (类似于 VEQ 中的 DP 过滤)
    # 确保在 1+k 步内，邻居的候选集仍然可达
    max_step = 1 + k
    changed = True
    while changed:
        changed = False
        for q in query_graph.nodes():
            new_set = set()
            neighbors = list(query_graph.neighbors(q))
            for d in cv_map[q]:
                # 检查 d 是否能到达每一个邻居的至少一个候选
                is_safe = True
                for q_n in neighbors:
                    # 寻找 d 附近 max_step 范围内的节点
                    found_neighbor_candidate = False
                    # 使用 BFS 检查可达性（带标签/度约束预判）
                    reachable = nx.single_source_shortest_path_length(cdg, d, cutoff=max_step)
                    if any(cand in reachable for cand in cv_map[q_n]):
                        found_neighbor_candidate = True
                    
                    if not found_neighbor_candidate:
                        is_safe = False
                        break
                if is_safe:
                    new_set.add(d)
            
            if len(new_set) < len(cv_map[q]):
                cv_map[q] = new_set
                changed = True
                if not cv_map[q]: return {}, {}, {}

    # 4. 路径预提取
    candidate_edge_map = defaultdict(set)
    candidate_path_dict = {}
    for q_u, q_v in query_graph.edges():
        q_key = tuple(sorted((q_u, q_v)))
        for d_u, d_v in product(cv_map[q_u], cv_map[q_v]):
            if d_u == d_v: continue
            d_pair = tuple(sorted((d_u, d_v)))
            if d_pair not in candidate_path_dict:
                # 获取所有 <= 1+k 的路径
                paths = list(nx.all_simple_paths(cdg, d_pair[0], d_pair[1], cutoff=max_step))
                candidate_path_dict[d_pair] = [(p, len(p)-1) for p in paths]
            
            if candidate_path_dict[d_pair]:
                min_l = min(p[1] for p in candidate_path_dict[d_pair])
                candidate_edge_map[q_key].add((d_u, d_v, min_l))

    return cv_map, dict(candidate_edge_map), candidate_path_dict

#基于路径安全的过滤策略
# def filter_by_global_weight_constraint(
#     query_graph: nx.Graph,
#     k: int,
#     candidate_vertex_map: dict,
#     candidate_edge_map: dict,
#     candidate_path_dict: dict
# ) -> tuple[dict, dict, dict]:
#     """
#     优化策略：
#     1. 粗过滤：基于点对的最短路径长度，剔除绝对不合规的数据对。
#     2. 细过滤：在保留的点对内部，精筛每一条路径，并利用连锁反应进行迭代。
#     """
#     print("--- 启动分层全局权重约束过滤 (先点对后路径) ---")
    
#     num_query_edges = query_graph.number_of_edges()
#     total_budget = num_query_edges + k
    
#     curr_vertex_map = {q_v: v_set.copy() for q_v, v_set in candidate_vertex_map.items()}
#     curr_edge_map = {q_e: e_set.copy() for q_e, e_set in candidate_edge_map.items()}
#     curr_path_dict = {pair: p_list.copy() for pair, p_list in candidate_path_dict.items()}

#     # ---------------------------------------------------------
#     # 步骤 A: 粗粒度过滤 (针对 candidate_edge_map)
#     # ---------------------------------------------------------
#     # 这一步只看点对的 w (即最短路径)，快速缩小范围
#     min_weights = {}
#     for q_e in query_graph.edges():
#         q_key = tuple(sorted(q_e))
#         if q_key not in curr_edge_map or not curr_edge_map[q_key]: return {}, {}, {}
#         min_weights[q_key] = min(info[2] for info in curr_edge_map[q_key])

#     total_min_sum = sum(min_weights.values())
    
#     # 粗过滤循环
#     for q_key, d_infos in list(curr_edge_map.items()):
#         # 过滤点对：如果点对最短路 + 其他边最小路 > 预算，整个点对滚蛋
#         allowed_w = total_budget - (total_min_sum - min_weights[q_key])
#         curr_edge_map[q_key] = {info for info in d_infos if info[2] <= allowed_w}
        
#         if not curr_edge_map[q_key]: return {}, {}, {}

#     # ---------------------------------------------------------
#     # 步骤 B: 细粒度迭代过滤 (针对 candidate_path_dict)
#     # ---------------------------------------------------------
#     # 在剩下的点对里，精筛每一条路径。因为路径被删可能导致 min_w 增加，所以需要迭代。
#     changed = True
#     while changed:
#         changed = False
        
#         # 1. 更新各边最小权重
#         min_weights = {q_k: min(info[2] for info in d_set) for q_k, d_set in curr_edge_map.items()}
#         total_min_sum = sum(min_weights.values())
#         if total_min_sum > total_budget: return {}, {}, {}

#         # 2. 建立点对到查询边的快速索引
#         pair_to_q_keys = defaultdict(set)
#         for q_k, d_infos in curr_edge_map.items():
#             for d_u, d_v, _ in d_infos:
#                 pair_to_q_keys[tuple(sorted((d_u, d_v)))].add(q_k)

#         # 3. 过滤具体路径
#         new_path_dict = {}
#         for d_pair, paths in curr_path_dict.items():
#             q_keys = pair_to_q_keys.get(d_pair)
#             if not q_keys: continue 
            
#             # 计算最大允许路径长度
#             max_limit = max(total_budget - (total_min_sum - min_weights[q_k]) for q_k in q_keys)
            
#             filtered = [p for p in paths if p[1] <= max_limit]
#             if len(filtered) < len(paths):
#                 changed = True
#                 if not filtered: continue # 该点对所有路径都阵亡了
#             new_path_dict[d_pair] = filtered

#         curr_path_dict = new_path_dict

#         # 4. 同步更新 edge_map
#         new_edge_map = defaultdict(set)
#         for q_k, d_infos in curr_edge_map.items():
#             for d_u, d_v, _ in d_infos:
#                 pair = tuple(sorted((d_u, d_v)))
#                 if pair in curr_path_dict:
#                     new_min_w = min(p[1] for p in curr_path_dict[pair])
#                     new_edge_map[q_k].add((d_u, d_v, new_min_w))
#             if not new_edge_map[q_k]: return {}, {}, {}
        
#         # 如果最短权重变了，changed 会在下一轮继续生效
#         if curr_edge_map != new_edge_map:
#             changed = True
#         curr_edge_map = dict(new_edge_map)

#     # ---------------------------------------------------------
#     # 步骤 C: 顶点清理
#     # ---------------------------------------------------------
#     active_nodes = {n for pair in curr_path_dict.keys() for n in pair}
#     final_vertex_map = {q_v: v_set.intersection(active_nodes) for q_v, v_set in curr_vertex_map.items()}
    
#     if any(not v_set for v_set in final_vertex_map.values()): return {}, {}, {}

#     print(f"--- 过滤完成: 剩余路径对={len(curr_path_dict)} ---")
#     return final_vertex_map, curr_edge_map, curr_path_dict

from bisect import bisect_right
def filter_by_global_weight_constraint(
    query_graph: nx.Graph,
    k: int,
    candidate_vertex_map: dict,
    candidate_edge_map: dict,
    candidate_path_dict: dict
) -> tuple[dict, dict, dict]:
    """
    高度优化的全局权重约束过滤算法。
    优化点：路径预排序、二分查找过滤、增量 Worklist 更新、最小权重缓存。
    """
    num_query_edges = query_graph.number_of_edges()
    total_budget = num_query_edges + k
    
    # 1. 初始化数据结构与预排序
    curr_vertex_map = {q_v: v_set.copy() for q_v, v_set in candidate_vertex_map.items()}
    # 路径按长度 (p[1]) 排序，以便后续二分查找
    sorted_path_dict = {}
    for pair, p_list in candidate_path_dict.items():
        sorted_path_dict[pair] = sorted(p_list, key=lambda x: x[1])

    # 2. 建立点对到查询边的静态索引，减少循环内计算
    pair_to_q_keys = defaultdict(list)
    curr_edge_map = {}
    for q_e in query_graph.edges():
        q_key = tuple(sorted(q_e))
        if q_key not in candidate_edge_map or not candidate_edge_map[q_key]:
            return {}, {}, {}
        
        # 只保留有对应路径的点对
        valid_infos = []
        for d_u, d_v, _ in candidate_edge_map[q_key]:
            pair = tuple(sorted((d_u, d_v)))
            if pair in sorted_path_dict:
                pair_to_q_keys[pair].append(q_key)
                # 初始权重取该点对路径的最小值
                min_w = sorted_path_dict[pair][0][1]
                valid_infos.append([d_u, d_v, min_w])
        
        if not valid_infos: return {}, {}, {}
        curr_edge_map[q_key] = valid_infos

    # 3. 计算初始状态
    q_min_w = {q_k: min(info[2] for info in infos) for q_k, infos in curr_edge_map.items()}
    total_min_sum = sum(q_min_w.values())
    if total_min_sum > total_budget: return {}, {}, {}

    # 4. 增量迭代过滤 (Worklist 机制)
    # 只要总预算不变，一旦某个 q_k 的最小权重增加，所有查询边都需要重新校对 max_limit
    worklist = deque(curr_edge_map.keys())
    in_queue = {q_k for q_k in curr_edge_map.keys()}

    while worklist:
        target_q_key = worklist.popleft()
        in_queue.remove(target_q_key)
        
        # 计算当前查询边允许的最大路径长度
        # 逻辑：budget - (所有边最小和 - 本边当前贡献的最小和)
        max_limit = total_budget - (total_min_sum - q_min_w[target_q_key])
        
        # 过滤该查询边下的所有点对
        new_infos = []
        q_min_changed = False
        
        for d_u, d_v, old_pair_min_w in curr_edge_map[target_q_key]:
            pair = tuple(sorted((d_u, d_v)))
            paths = sorted_path_dict[pair]
            
            # 使用二分查找快速找到符合长度限制的路径切片位置
            # paths 是按 p[1] (长度) 排序的
            idx = bisect_right(paths, max_limit, key=lambda x: x[1])
            
            if idx > 0:
                # 仍然存在合法路径
                new_pair_min_w = paths[0][1] # 排序后的第一个始终是该点对的最小权重
                new_infos.append([d_u, d_v, new_pair_min_w])
                
                # 如果该点对被截断了（虽然最小权重没变，但长路径没了），
                # 这里我们只关注最小权重变化，因为只有它会影响其他边的 max_limit
                if len(paths) > idx:
                    sorted_path_dict[pair] = paths[:idx]
            else:
                # 该点对所有路径都阵亡了
                del sorted_path_dict[pair]

        if len(new_infos) < len(curr_edge_map[target_q_key]):
            if not new_infos: return {}, {}, {}
            
            curr_edge_map[target_q_key] = new_infos
            new_q_min = min(info[2] for info in new_infos)
            
            # 如果该查询边的最小权重增加了，会挤压其他所有边的预算空间
            if new_q_min > q_min_w[target_q_key]:
                total_min_sum += (new_q_min - q_min_w[target_q_key])
                q_min_w[target_q_key] = new_q_min
                
                # 将其他所有边加入待检查队列（因为它们的 max_limit 变小了）
                for other_q in curr_edge_map.keys():
                    if other_q != target_q_key and other_q not in in_queue:
                        worklist.append(other_q)
                        in_queue.add(other_q)

    # 5. 最终清理与输出格式转换
    final_path_dict = {pair: p_list for pair, p_list in sorted_path_dict.items() if pair in pair_to_q_keys}
    final_edge_map = {q_k: set(tuple(info) for info in infos) for q_k, infos in curr_edge_map.items()}
    
    active_nodes = {n for pair in final_path_dict.keys() for n in pair}
    final_vertex_map = {q_v: v_set.intersection(active_nodes) for q_v, v_set in curr_vertex_map.items()}
    
    if any(not v_set for v_set in final_vertex_map.values()): return {}, {}, {}

    return final_vertex_map, final_edge_map, final_path_dict

#验证与回溯
# def validate_and_find_matches(
#     query_graph: nx.Graph,
#     k: int,
#     candidate_vertex_map: dict,
#     final_path_dict: dict,  # 新增：预过滤后的路径字典
# ) -> list:
#     """
#     重构后的验证函数：基于预存路径字典进行极速回溯。
#     """
    
    
#     track_stats = {
#         'total_path_calls': 0,
#         'conflict_backtracks': 0,
#         'found_count': 0
#     }
    
#     # --- 1. 基础参数准备 ---
#     verified_subgraphs = []
#     # 获取查询边列表，后续会根据映射动态排序
#     raw_query_edges = list(query_graph.edges())
#     num_query_edges = len(raw_query_edges)
#     total_budget = num_query_edges + k
    
#     # 顶点回溯顺序：优先处理候选集小的顶点
#     sorted_query_nodes = sorted(
#         query_graph.nodes(),
#         key=lambda n: len(candidate_vertex_map.get(n, []))
#     )

#     # =================================================================
#     # 2. 内部路径回溯函数 (基于缓存)
#     # =================================================================

#     def _backtrack_paths(edge_idx, ordered_edges, mapping, current_paths, blocked_internal_nodes, current_total_weight):
#         track_stats['total_path_calls'] += 1
        
#         # --- 基线条件：所有边处理完毕 ---
#         if edge_idx == num_query_edges:
#             result_graph = nx.Graph()
#             for path in current_paths.values():
#                 nx.add_path(result_graph, path)
#             verified_subgraphs.append(result_graph)
#             track_stats['found_count'] += 1
#             return

#         # --- 动态获取当前边的候选路径 ---
#         q_u, q_v = ordered_edges[edge_idx]
#         d_u, d_v = mapping[q_u], mapping[q_v]
#         d_pair = tuple(sorted((d_u, d_v)))
        
#         # 从预存储字典中直接获取路径列表 (path, length)
#         possible_paths = final_path_dict.get(d_pair, [])
        
#         # --- 预算与屏蔽集合准备 ---
#         remaining_edges = num_query_edges - edge_idx
#         all_anchors = set(mapping.values())
#         # 屏蔽点 = (所有锚点 + 已占用内部节点) - (当前路径两端)
#         base_blocked = (all_anchors | blocked_internal_nodes) - {d_u, d_v}
        
#         # 允许的最大长度
#         max_len_allowed = total_budget - current_total_weight - (remaining_edges - 1)

#         # --- 遍历预存路径 ---
#         for path, path_len in possible_paths:
#             # 1. 长度检查 (虽然 final_path_dict 已过滤，但此处需配合全局动态预算)
#             if path_len > max_len_allowed:
#                 continue
            
#             # 2. 冲突检查 (Disjointness Check)
#             internal_nodes = set(path[1:-1])
#             if not internal_nodes.isdisjoint(base_blocked):
#                 track_stats['conflict_backtracks'] += 1
#                 continue

#             # 3. 递归递归
#             edge_key = tuple(sorted((q_u, q_v)))
#             current_paths[edge_key] = path
            
#             _backtrack_paths(
#                 edge_idx + 1,
#                 ordered_edges,
#                 mapping,
#                 current_paths,
#                 blocked_internal_nodes | internal_nodes,
#                 current_total_weight + path_len
#             )
            
#             # 回溯
#             del current_paths[edge_key]

#     # =================================================================
#     # 3. 内部顶点回溯函数 (触发路径搜索)
#     # =================================================================

#     def _backtrack_vertices(node_idx, current_mapping):
#         if node_idx == len(sorted_query_nodes):
#             # --- 关键优化：MRV 路径定序 ---
#             # 在进入路径搜索前，根据当前的顶点映射，对边进行排序
#             # 优先处理候选路径最少的查询边
#             edge_potential = []
#             for q_u, q_v in raw_query_edges:
#                 d_u, d_v = current_mapping[q_u], current_mapping[q_v]
#                 d_pair = tuple(sorted((d_u, d_v)))
#                 count = len(final_path_dict.get(d_pair, []))
#                 edge_potential.append(((q_u, q_v), count))
            
#             # 按候选数量升序排序
#             ordered_edges = [e for e, c in sorted(edge_potential, key=lambda x: x[1])]
            
#             # 如果某条边完全没路，直接剪枝
#             if any(c == 0 for e, c in edge_potential):
#                 return

#             _backtrack_paths(0, ordered_edges, current_mapping, {}, set(), 0)
#             return

#         q_node = sorted_query_nodes[node_idx]
#         candidates = candidate_vertex_map.get(q_node, set())
#         used_data_nodes = set(current_mapping.values())

#         for d_node in candidates:
#             if d_node in used_data_nodes:
#                 continue
            
#             # 连通性预检查
#             is_valid = True
#             for neighbor in query_graph.neighbors(q_node):
#                 if neighbor in current_mapping:
#                     d_neighbor = current_mapping[neighbor]
#                     d_pair = tuple(sorted((d_node, d_neighbor)))
#                     # 如果预存字典里根本没这对点的路径，说明映射非法
#                     if d_pair not in final_path_dict or not final_path_dict[d_pair]:
#                         is_valid = False
#                         break
            
#             if is_valid:
#                 current_mapping[q_node] = d_node
#                 _backtrack_vertices(node_idx + 1, current_mapping)
#                 del current_mapping[q_node]

#     # =================================================================
#     # 4. 执行
#     # =================================================================
#     _backtrack_vertices(0, {})
    
#     print(f"--- 验证结束: 递归总数={track_stats['total_path_calls']}, "
#           f"找到解={len(verified_subgraphs)}, 冲突拦截={track_stats['conflict_backtracks']} ---")
    
    
#     return verified_subgraphs

import networkx as nx
from collections import defaultdict, deque
from itertools import product

import networkx as nx

def validate_and_find_matches(
    query_graph: nx.Graph,
    k: int,
    candidate_vertex_map: dict,
    final_path_dict: dict
) -> list:
    """
    修正后的同胚匹配验证算法：
    修复了路径路由时的贪心截断问题，确保枚举所有符合全局预算的路径组合。
    并精准生成与细分查询图 (VF2) 格式完全一致的虚拟节点字典映射。
    """
    verified_subgraphs = []
    num_query_edges = query_graph.number_of_edges()
    total_max_weight = num_query_edges + k
    
    # 1. 预处理：将路径按长度升序排列，方便预算剪枝
    sorted_path_dict = {}
    for k_pair, paths in final_path_dict.items():
        sorted_path_dict[k_pair] = sorted(paths, key=lambda x: x[1])
        
    # 2. 确定匹配序：度数大优先，候选集小优先
    ordered_nodes = sorted(
        query_graph.nodes(), 
        key=lambda n: (-query_graph.degree(n), len(candidate_vertex_map.get(n, [])))
    )
    
    # 3. 记录原始查询图边的顺序方向，用于规范化输出虚拟节点的名称，对齐 VF2
    original_edges = list(query_graph.edges())
    edge_directions = {tuple(sorted((u, v))): (u, v) for u, v in original_edges}

    def backtrack(node_idx, mapping, used_d_nodes, current_weight, edge_paths):
        if node_idx == len(ordered_nodes):
            # 此时锚点映射完成，所有路径就位。构造和 VF2 完全一致的 full_map
            full_map = mapping.copy()
            for q_edge_key, path in edge_paths.items():
                if len(path) > 2:  # 只有长度 > 1 的路径才有内部(虚拟)节点
                    u, v = edge_directions[q_edge_key]
                    # 根据原始边的有向性决定内部节点的命名顺序
                    if path[0] == mapping[u]:
                        internal = path[1:-1]
                    else:
                        internal = path[-2:0:-1]
                        
                    for v_idx, d_node in enumerate(internal):
                        v_name = f"v_{u}_{v}_{v_idx}"
                        full_map[v_name] = d_node
                        
            verified_subgraphs.append(full_map)
            return

        q_u = ordered_nodes[node_idx]
        matched_neighbors = [n for n in query_graph.neighbors(q_u) if n in mapping]
        
        for d_u in candidate_vertex_map.get(q_u, []):
            if d_u in used_d_nodes:
                continue

            # 收集当前锚点 d_u 与所有已匹配邻居的预存合法路径
            valid_paths_per_neighbor = []
            is_feasible = True
            for q_v in matched_neighbors:
                d_v = mapping[q_v]
                d_pair = tuple(sorted((d_u, d_v)))
                paths = sorted_path_dict.get(d_pair, [])
                if not paths:
                    is_feasible = False
                    break
                valid_paths_per_neighbor.append((q_v, paths))
            
            if not is_feasible:
                continue

            # --- 核心修复：嵌套 DFS 以枚举当前节点相邻边的所有合法路径组合 ---
            def route_edges(neighbor_idx, current_w, temp_used, temp_edge_paths):
                if neighbor_idx == len(valid_paths_per_neighbor):
                    # 当前节点 q_u 的所有相邻边都成功分配了不冲突的路径
                    mapping[q_u] = d_u
                    used_d_nodes.add(d_u)
                    used_d_nodes.update(temp_used)
                    for k_edge, p in temp_edge_paths.items():
                        edge_paths[k_edge] = p
                        
                    # 递归匹配下一个查询锚点
                    backtrack(node_idx + 1, mapping, used_d_nodes, current_w, edge_paths)
                    
                    # 回溯清理锚点与路径状态
                    for k_edge in temp_edge_paths:
                        del edge_paths[k_edge]
                    used_d_nodes.difference_update(temp_used)
                    used_d_nodes.remove(d_u)
                    del mapping[q_u]
                    return

                q_v, possible_paths = valid_paths_per_neighbor[neighbor_idx]
                
                # 全局预算动态感知：(总边数 - 当前已匹配及即将匹配的边数) * 1 (最小开销)
                unmatched_edges = num_query_edges - (len(edge_paths) + neighbor_idx)
                remaining_min_future = max(0, unmatched_edges - 1)
                
                for path, p_len in possible_paths:
                    # 预算剪枝：如果超限，因为 paths 已按长度升序，后续的必然也超标，直接 break
                    if current_w + p_len + remaining_min_future > total_max_weight:
                        break 
                    
                    internal_nodes = set(path[1:-1])
                    # 独立路径检查：不能碰撞历史被用的节点，也不能碰撞本次分配同源的节点
                    if internal_nodes.isdisjoint(used_d_nodes) and internal_nodes.isdisjoint(temp_used):
                        temp_used.update(internal_nodes)
                        q_edge_key = tuple(sorted((q_u, q_v)))
                        temp_edge_paths[q_edge_key] = path
                        
                        # 继续分配下一条相连的边 (无 break，意味着将穷尽这所有的路由可能)
                        route_edges(neighbor_idx + 1, current_w + p_len, temp_used, temp_edge_paths)
                        
                        # 回溯清理本条边占用的节点
                        temp_used.difference_update(internal_nodes)
                        del temp_edge_paths[q_edge_key]

            # 启动当前锚点的边路由分发
            route_edges(0, current_weight, set(), {})

    # 启动算法
    backtrack(0, {}, set(), 0, {})
    return verified_subgraphs


#组合函数
def ppsm(query_graph: nx.Graph,
    k: int,
    data_graph: nx.Graph,
    )-> list:
    
    #数据图预处理
    bridges, components = custom_graph_partition(data_graph)
    subgraph_map, total_tutte = tutte_from_bridge_decomposition_parallel(bridges, components)
    block_cut_tree = build_block_cut_tree(bridges, subgraph_map)
    tutte_index = build_rect_dominance_index_keep_self(subgraph_map)
    
    i_set = {i for i, _, _ in subgraph_map}
    
    
    q_tutte = approx_tutte_rectangle_maxonly(query_graph)
    #通过多项式覆盖初步估计是否存在解
    if 1==1:
        #查询处理
        #candidate_set=dispatch_query_processing(query_graph, subgraph_map, block_cut_tree, tutte_index,k,bridges)
        #构建候选解空间
        candidate_vertex_map, candidate_edge_set, candidate_path_dict = construct_candidate_solution_space(query_graph,k,i_set,subgraph_map,bridges)
        #过滤策略
        f_candidate_vertex_map, f_candidate_edge_set, f_path_dict = filter_by_global_weight_constraint(query_graph,k,candidate_vertex_map,candidate_edge_set,candidate_path_dict)
        #验证与回溯
        matchs = validate_and_find_matches(query_graph,k,f_candidate_vertex_map,f_path_dict)
        return matchs
    else:
        print("没有匹配")
        return []

import time

def main():
    query_graph = nx.Graph([(0,1), (0,2), (1,3), (2,4), (3,5)])
    file_path = "./graphs/ws_Et20_Er70_n35.txt" 
    data_graph = load_graph_from_txt(file_path)
    t0= time.time()
    matches = ppsm(query_graph,2,data_graph)
    t1= time.time()
    print("run time:",t1-t0)
    print(len(matches))

    # unique_subgraphs = {}

    # for g in matches:
    #     key = frozenset(tuple(sorted(e)) for e in g.edges())
    #     if key not in unique_subgraphs:
    #         unique_subgraphs[key] = g

    # print("\nUnique subgraph count:", len(unique_subgraphs))


    # if matches:
    #     g_union = nx.compose_all(matches)
    # else:
    #     g_union = nx.Graph()
    # print(g_union.nodes)
    # print(g_union.edges)
    
if __name__ == "__main__":
    main()
        
