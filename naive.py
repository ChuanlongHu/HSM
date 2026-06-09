#原始方法
import networkx as nx
import itertools
from typing import List, Tuple, Dict, Set, Any

# ---------------------- 辅助：枚举所有 simple paths（朴素 DFS） ----------------------
def all_simple_paths_dfs(G: nx.Graph, source: Any, target: Any):
    """
    生成图 G 中从 source 到 target 的所有简单路径（朴素 DFS 枚举）。
    注意：对较大图/远距离节点会非常多。
    """
    if source == target:
        yield [source]
        return
    visited = set([source])
    stack = [(source, iter(G.neighbors(source)), [source])]
    while stack:
        node, nbrs_iter, path = stack[-1]
        try:
            nb = next(nbrs_iter)
            if nb in visited:
                continue
            if nb == target:
                yield path + [nb]
            else:
                visited.add(nb)
                stack.append((nb, iter(G.neighbors(nb)), path + [nb]))
        except StopIteration:
            stack.pop()
            visited.discard(node)

# ---------------------- 1) 枚举子图：边数在 [min_e, max_e] 的所有 edge-induced 子图 ----------------------
def enumerate_edge_subgraphs(G: nx.Graph, min_e: int, max_e: int):
    """
    枚举 G 上所有 edge-induced 子图 H，它们的边数 e 满足 min_e <= e <= max_e。
    返回 generator，每个元素为 (edge_set_frozenset, H_graph)
      - edge_set_frozenset: frozenset of edge tuples (u,v) with the same ordering as G.edges()
      - H_graph: networkx.Graph induced by those edges (copy)
    警告：组合数巨大，谨慎调用。
    """
    edges = list(G.edges())
    n = len(edges)
    if min_e < 0: min_e = 0
    if max_e > n: max_e = n
    # iterate sizes
    for r in range(min_e, max_e + 1):
        for comb in itertools.combinations(range(n), r):
            sel_edges = [edges[i] for i in comb]
            H = G.edge_subgraph(sel_edges).copy()
            if not nx.is_connected(H):
                continue
            yield frozenset(sel_edges), H

# ---------------------- 2) 枚举 Q 在 H 上的所有同胚映射（完全枚举） ----------------------
def enumerate_homeomorphic_mappings(Q: nx.Graph, H: nx.Graph):
    """
    枚举 Q 在 H 上的所有同胚映射（topological embeddings）。
    返回 mappings: list of mapping records, 每个 record 为 dict:
        {
          "vertex_map": {q_node: h_node, ...},           # injective mapping
          "edge_paths": { (u,v): [node_list_path], ... } # 对每条 Q 边给出一条 H 上的简单路径
        }
    说明（暴力做法）：
      - 枚举所有 injective mappings φ: V(Q) -> V(H)（通过 permutations）
      - 对每个映射，枚举每条 Q 边在 H 中所有可能的 simple paths（内部节点不能是映像点）
      - 选择一组路径要求：不同 Q 边对应路径的内部节点两两不相交
      - 对每个满足条件的组合，保存为一个映射记录
    注意：返回集合可能非常大。
    """
    mappings = []
    nQ = Q.number_of_nodes()
    if nQ == 0:
        # empty mapping
        mappings.append({"vertex_map": {}, "edge_paths": {}})
        return mappings

    H_nodes = list(H.nodes())
    Q_nodes = list(Q.nodes())
    Q_edges = list(Q.edges())

    # quick trivial rejects
    if nQ > H.number_of_nodes() or Q.number_of_edges() > H.number_of_edges():
        return mappings

    # iterate all injective mappings (permutations)
    for perm in itertools.permutations(H_nodes, nQ):
        vertex_map = {Q_nodes[i]: perm[i] for i in range(nQ)}
        images = set(vertex_map.values())
        # basic degree feasibility check
        feasible = True
        for qv in Q_nodes:
            if H.degree(vertex_map[qv]) < Q.degree(qv):
                feasible = False
                break
        if not feasible:
            continue

        # Now enumerate choices of simple paths for each Q edge with backtracking
        edge_paths_chosen = {}   # map edge -> chosen path (list)
        used_internal = set()    # internal nodes already used by earlier-chosen edge paths
        # define recursion over edges (use Q_edges order)
        def backtrack_edge(idx):
            if idx == len(Q_edges):
                # found one full set of paths -> record mapping
                mappings.append({
                    "vertex_map": dict(vertex_map),
                    "edge_paths": dict(edge_paths_chosen)
                })
                return

            u, v = Q_edges[idx]
            s = vertex_map[u]
            t = vertex_map[v]

            # enumerate all simple paths s->t in H
            for path in all_simple_paths_dfs(H, s, t):
                internal = path[1:-1]
                # internal nodes cannot include any image nodes
                conflict = False
                for w in internal:
                    if w in images or w in used_internal:
                        conflict = True
                        break
                if conflict:
                    continue
                # accept this path and recurse
                edge_paths_chosen[(u, v)] = list(path)
                for w in internal:
                    used_internal.add(w)
                backtrack_edge(idx + 1)
                # backtrack
                for w in internal:
                    used_internal.remove(w)
                del edge_paths_chosen[(u, v)]
            # end for paths
            return

        backtrack_edge(0)

    return mappings

# ---------------------- 3) 主流程：枚举 H（边数在 |E(Q)|..|E(Q)|+k），统计同胚子图数量 ----------------------
def count_homeomorphic_subgraphs(
    G: nx.Graph,
    Q: nx.Graph,
    k: int,
    verbose: bool = False,
    max_enumerated_subgraphs: int = None
):
    """
    修改版本：
      - count 变量现在记录的是所有有效映射的总数，而不仅仅是子图的个数。
    """
    min_e = Q.number_of_edges()
    max_e = Q.number_of_edges() + k

    matched_map = {}
    total_enumerated = 0
    total_mappings_count = 0  # 初始化总映射计数器

    for edge_set, H in enumerate_edge_subgraphs(G, min_e, max_e):
        total_enumerated += 1

        if max_enumerated_subgraphs is not None and total_enumerated > max_enumerated_subgraphs:
            if verbose:
                print(f"早停：已枚举 {total_enumerated} 个子图")
            break

        if H.number_of_nodes() < Q.number_of_nodes():
            continue

        mappings = enumerate_homeomorphic_mappings(Q, H)
        if not mappings:
            continue

        H_edge_set = set(tuple(sorted(e)) for e in H.edges())
        valid_mappings = []

        for m in mappings:
            edge_paths = m["edge_paths"]
            used_edges = set()
            for path in edge_paths.values():
                for u, v in zip(path[:-1], path[1:]):
                    used_edges.add(tuple(sorted((u, v))))

            # 关键过滤条件：映射所使用的边必须正好等于 H 的边集
            if used_edges == H_edge_set:
                valid_mappings.append(m)

        if valid_mappings:
            matched_map[edge_set] = valid_mappings
            # --- 核心修改点：累加当前子图发现的有效映射数量 ---
            total_mappings_count += len(valid_mappings)
            
            if verbose:
                print(
                    f"匹配: H edges={sorted(edge_set)} "
                    f"当前子图映射数={len(valid_mappings)}, 累计总映射数={total_mappings_count}"
                )

    # 返回总映射数和详细字典
    return total_mappings_count, matched_map, total_enumerated


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

def main():
    query_graph = nx.Graph([(0, 1), (1, 2), (2, 0), (1,3), (2,3)])
    file_path = "brige_test.txt" 
    data_graph = load_graph_from_txt(file_path)
    count, matches, subgraphs = count_homeomorphic_subgraphs(data_graph,query_graph,2,verbose=False)
    for i, edge_set in enumerate(matches.keys(), 1):
        print(f"子图 {i}:")
        print(sorted(edge_set))
    print("sugraphs",subgraphs)
    print("matchedgraphs",len(matches))
    print("count",count)
    
if __name__ == "__main__":
    main()
        
