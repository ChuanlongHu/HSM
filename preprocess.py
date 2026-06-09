#预处理过程测试
import os
import json
import time
import ppsm
import networkx as nx
import sys


sys.setrecursionlimit(20000)


# ===============================
# 图结构序列化 / 反序列化
# ===============================
def nx_to_dict(G):
    return {
        "nodes": list(G.nodes()),
        # 修正：将边 (元组) 转换为列表的列表，提高 JSON 兼容性
        "edges": [list(edge) for edge in G.edges()] 
    }

def dict_to_nx(d):
    G = nx.Graph()
    G.add_nodes_from(d["nodes"])
    # 反序列化时，将列表转换回元组
    G.add_edges_from([tuple(edge) for edge in d["edges"]]) 
    return G


# ===============================
# 根据 txt 文件生成 JSON 名称
# ===============================
def get_json_name_from_txt(txt_path: str) -> str:
    base = os.path.splitext(os.path.basename(txt_path))[0]
    return f"{base}_preprocessed.json"


# ===============================
# 读取 txt 数据图
# ===============================
def load_graph_from_txt(file_path):
    G = nx.Graph()
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                u, v = map(int, line.split())
                G.add_edge(u, v)
    return G


def serialize_subgraph_map(subgraph_map):
    result = []
    for (gid, g, tutte_dict) in subgraph_map:
        result.append({
            "id": gid,
            "graph": nx_to_dict(g),
            # 修正：确保 tutte_dict 的键是字符串
            "tutte": {str(k): v for k, v in tutte_dict.items()} 
        })
    return result


def ensure_json_compatible_data(data):
    """
    递归地将数据结构中的集合(set)转换为列表(list)，
    并将所有字典键转换为字符串(str)。
    """
    if isinstance(data, set):
        # 将集合转换为列表 (解决 set value 错误)
        return [ensure_json_compatible_data(item) for item in data]
    elif isinstance(data, dict):
        # 递归处理字典：同时确保键是字符串 (解决 tuple key 错误)
        # 并且值也兼容 JSON
        return {str(k): ensure_json_compatible_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        # 递归处理列表中的元素
        return [ensure_json_compatible_data(item) for item in data]
    elif isinstance(data, tuple):
        # 如果元组是列表中的值，将其转换为列表，以提高兼容性
        return [ensure_json_compatible_data(item) for item in data]
    else:
        # 其他类型保持不变
        return data

# ===============================
# 保存预处理结果
# ===============================
def save_preprocessing_result_auto(
    txt_path,
    bridges, components, subgraph_map, total_tutte, block_cut_tree, tutte_index,
    time_logs
):
    json_path = get_json_name_from_txt(txt_path)

    tutte_index_compatible = ensure_json_compatible_data(tutte_index)
    total_tutte_compatible = ensure_json_compatible_data(total_tutte)

    # 修正 3：将 bridges (set of tuples) 转换为 list of lists
    bridges_list_of_lists = [list(edge) for edge in bridges]

    data = {
        "bridges": bridges_list_of_lists,
        # 修正 4：确保 components 中的每个 Graph 被 nx_to_dict 序列化
        "components": [nx_to_dict(g) for g in components],
        "total_tutte": total_tutte_compatible, 
        "tutte_index": tutte_index_compatible, 
        "time_logs": time_logs,
        "subgraph_map": serialize_subgraph_map(subgraph_map),
        "block_cut_tree": nx_to_dict(block_cut_tree)
    }

    with open(json_path, "w") as f:
        json.dump(data, f)

    print(f"\n预处理结果已保存到：{json_path}")


# ===============================
# 预处理主函数（可独立测试）
# ===============================
def preprocess_graph(txt_path):
    print(f"开始预处理：{txt_path}")

    # --- 加载数据图 ---
    t0 = time.time()
    data_graph = load_graph_from_txt(txt_path)
    t1 = time.time()
    print(f"加载数据图耗时：{t1 - t0:.3f} 秒")

    time_logs = {
        "load_graph": t1 - t0
    }

    # --- 步骤 1：Bridge 分解 ---
    t0 = time.time()
    bridges, components = ppsm.custom_graph_partition(data_graph) 
    t1 = time.time()
    print(f"步骤1 custom_graph_partition 耗时：{t1 - t0:.3f} 秒")
    print("分量数量",len(components))
    time_logs["custom_graph_partition"] = t1 - t0

    # --- 步骤 2：Tutte 多项式分治计算 ---
    t0 = time.time()
    subgraph_map, total_tutte = ppsm.tutte_from_bridge_decomposition_parallel(bridges, components)
    t1 = time.time()
    print(f"步骤2 tutte_from_bridge_decomposition_parallel 耗时：{t1 - t0:.3f} 秒")
    time_logs["tutte_from_bridge_decomposition_parallel"] = t1 - t0

    # --- 步骤 3：构建 Block-Cut Tree ---
    t0 = time.time()
    block_cut_tree = ppsm.build_block_cut_tree(bridges, subgraph_map)
    t1 = time.time()
    print(f"步骤3 build_block_cut_tree 耗时：{t1 - t0:.3f} 秒")
    time_logs["build_block_cut_tree"] = t1 - t0

    # --- 步骤 4：构建 Tutte 索引 ---
    t0 = time.time()
    tutte_index = ppsm.build_rect_dominance_index_keep_self(subgraph_map)
    t1 = time.time()
    print(f"步骤4 build_rect_dominance_index_keep_self 耗时：{t1 - t0:.3f} 秒")
    time_logs["build_rect_dominance_index_keep_self"] = t1 - t0
    

    # --- 保存结果 ---
    save_preprocessing_result_auto(
        txt_path,
        bridges, components, subgraph_map, total_tutte, block_cut_tree, tutte_index,
        time_logs
    )


# ===============================
# 主入口
# ===============================
if __name__ == "__main__":
    data_file = "./data/ca-AstroPh.txt"
    preprocess_graph(data_file)
