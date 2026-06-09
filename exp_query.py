#expand query
import time
import networkx as nx
from itertools import product
from networkx.algorithms import isomorphism

def load_graph_from_txt(file_path):
    G = nx.Graph()
    try:
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    G.add_edge(int(parts[0]), int(parts[1]))
    except FileNotFoundError:
        # 仅用于演示，如果没有文件则返回空图
        return nx.Graph()
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def generate_all_subdivisions(query_graph, k_max):
    """
    生成所有边细分总数从 0 到 k_max 的扩展图。
    每个扩展图通过其边分配方案进行唯一标识。
    """
    edges = list(query_graph.edges())
    num_edges = len(edges)
    all_expanded_graphs = []

    # 遍历所有可能的细分总数 i (0 <= i <= k_max)
    for i in range(k_max + 1):
        # 使用 product 生成所有可能的分配方案，确保 (1,0,0), (0,1,0), (0,0,1) 都被视为独立情况
        # 即使它们产生的图结构相似，但在同胚映射中它们代表不同的边被细分
        for p_dist in product(range(i + 1), repeat=num_edges):
            if sum(p_dist) == i:
                new_q = nx.Graph()
                # 标记原始节点集合
                new_q.graph['orig_nodes'] = set(query_graph.nodes())
                new_q.graph['dist_scheme'] = p_dist  # 记录分配方案方便调试
                
                for idx, (u, v) in enumerate(edges):
                    num_v_nodes = p_dist[idx]
                    if num_v_nodes == 0:
                        new_q.add_edge(u, v)
                    else:
                        # 插入虚拟顶点，命名必须包含边信息，防止多边细分时节点冲突
                        last_node = u
                        for v_idx in range(num_v_nodes):
                            v_node = f"v_{u}_{v}_{v_idx}"
                            new_q.add_edge(last_node, v_node)
                            last_node = v_node
                        new_q.add_edge(last_node, v)
                all_expanded_graphs.append(new_q)
    
    return all_expanded_graphs

def match_homeomorphic(query_graph, data_graph, k):
    """
    通过 VF2 算法收集所有同胚映射。
    """
    # 1. 生成所有扩展图（包含 k=0, k=1...k）
    expanded_queries = generate_all_subdivisions(query_graph, k)
    print(f"生成的扩展图数量 (含原始图): {len(expanded_queries)}")
    
    final_homeomorphic_mappings = []

    for eq in expanded_queries:
        # 2. 调用 VF2 寻找子图同构
        matcher = isomorphism.GraphMatcher(data_graph, eq)
        
        # 3. 收集映射
        # 注意：这里我们保留 eq_node -> data_node 的完整映射
        # 如果你只想要原始节点的映射，可以后续提取
        for iso in matcher.subgraph_monomorphisms_iter():
            # iso 默认是 {data_node: eq_node}，转换回 {query_node: data_node}
            full_map = {v: k for k, v in iso.items()}
            final_homeomorphic_mappings.append(full_map)
                
    return final_homeomorphic_mappings


if __name__ == "__main__":
    Q = nx.Graph([(0,1), (0,2),(1,2), (1,3), (2,3)])
    
    file_path = "./bio-CE-HT.txt" 
    G = load_graph_from_txt(file_path)
    
    k_val = 3
    t0= time.time()
    results = match_homeomorphic(Q, G, k_val)
    t1= time.time()
    print("run time:",t1 - t0)
    print(f"找到 {len(results)} 个独特的同胚映射:")
    # for idx, mapping in enumerate(results):
    #     print(f"映射 {idx + 1}: {mapping}")