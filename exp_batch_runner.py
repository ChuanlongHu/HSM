import networkx as nx
import time
import json
import os
import signal

# 导入扩展算法函数
from exp_query import match_homeomorphic, generate_all_subdivisions

# =====================================
# 超时处理 (Linux/Unix)
# =====================================
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# =====================================
# 数据加载与路径处理
# =====================================
def load_single_graph(path):
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            u, v = line.split()
            G.add_edge(int(u), int(v))
    return G

def load_query_graphs(path):
    graphs = []
    edges = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if edges:
                    graphs.append(nx.Graph(edges))
                    edges = []
                continue
            u, v = line.split()
            edges.append((int(u), int(v)))
    if edges:
        graphs.append(nx.Graph(edges))
    return graphs

def get_prefix(path):
    # 获取不带后缀的文件名
    return os.path.splitext(os.path.basename(path))[0]

def build_output_path_info(query_file, data_file, k, timeout):
    """
    根据要求构造路径：./results/k{k}_t{timeout}/数据图文件名/查询图文件名_exp.json
    """
    data_prefix = get_prefix(data_file)
    query_prefix = get_prefix(query_file)
    
    # 1. 根目录与参数目录
    kt_dir = f"k{k}_t{timeout}"
    base_dir = os.path.join("results", kt_dir, data_prefix)
    
    # 2. 完整文件路径
    file_name = f"{query_prefix}_exp.json"
    full_path = os.path.join(base_dir, file_name)
    
    return base_dir, full_path

# =====================================
# 单个查询运行逻辑
# =====================================
def run_single_query_baseline(G_data, G_query, k, timeout):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    start_time = time.time()
    num_variants = 0
    
    try:
        # 生成变体（仅用于统计基准复杂度）
        variants = generate_all_subdivisions(G_query, k)
        num_variants = len(variants)
        
        # 执行匹配
        mappings = match_homeomorphic(G_query, G_data, k)
        
        elapsed = time.time() - start_time
        signal.alarm(0)
        return "ok", len(mappings), elapsed, num_variants

    except TimeoutException:
        return "timeout", None, float(timeout), num_variants
    except Exception as e:
        signal.alarm(0)
        return "error", None, str(e), num_variants

# =====================================
# 批处理主函数
# =====================================
def batch_run(query_file, data_file, k, timeout):
    # 构造输出路径
    output_dir, out_path = build_output_path_info(query_file, data_file, k, timeout)
    os.makedirs(output_dir, exist_ok=True)

    G_data = load_single_graph(data_file)
    queries = load_query_graphs(query_file)

    results_data = {
        "query_file": query_file,
        "data_file": data_file,
        "k": k,
        "timeout": timeout,
        "results": []
    }

    print(f"Target: {out_path}")

    for i, Q in enumerate(queries):
        print(f"  Query {i}...", end="", flush=True)
        status, count, info, n_variants = run_single_query_baseline(G_data, Q, k, timeout)
        
        rec = {
            "query_id": i,
            "status": status,
            "time": info if status in ["ok", "timeout"] else None,
            "match_count": count,
            "expanded_variants": n_variants
        }
        results_data["results"].append(rec)
        print(f" {status} ({info if isinstance(info, float) else 'err':.2f}s)")

    # 写入 JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
    print(f"Successfully saved to {out_path}\n")

# =====================================
# 实验执行入口
# =====================================
if __name__ == "__main__":
    # 数据图路径
    DATA_FILES = ["./bio-CE-HT.txt"]
    
    # 查询图路径
    #QUERY_FILES=["./queries/bfree_8.txt","./queries/bfree_16.txt","./queries/bfree_32.txt", "./queries/tree_8.txt","./queries/tree_16.txt","./queries/tree_32.txt", "./queries/mixed_8.txt","./queries/mixed_16.txt","./queries/mixed_32.txt"]
    QUERY_FILES = [
        # "./queries/all_connected_E3.txt",
        # "./queries/all_connected_E4.txt",
        "./queries/all_connected_E5.txt"
    ]
    
    K_LIST = [1,2,3,4]
    TIMEOUT = 600 # 对应路径中的 t600

    for qf in QUERY_FILES:
        for df in DATA_FILES:
            for k in K_LIST:
                if not (os.path.exists(qf) and os.path.exists(df)):
                    continue
                
                print("=" * 60)
                batch_run(qf, df, k, TIMEOUT)