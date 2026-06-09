# ppsm_batch_runner.py
import ppsm
import time
import json
import os
from datetime import datetime
from typing import Callable, List, Tuple, Dict, Any, Optional

import networkx as nx
import signal
import glob


# ---------------------------
# Timeout helpers (Unix)
# ---------------------------
class QueryTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise QueryTimeout()


# ---------------------------
# IO helpers
# ---------------------------
def load_graph_from_txt(file_path: str) -> nx.Graph:
    G = nx.Graph()
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                u, v = map(int, line.split())
                G.add_edge(u, v)
    return G


def load_query_graphs_from_txt(path: str) -> List[nx.Graph]:
    """
    从文本文件读取一组图。每行一个边 "u v"（由空格分隔）。
    图与图之间通过空行分隔。
    返回 networkx.Graph 的列表。
    """
    graphs = []
    current_edges = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line == "":
                if current_edges:
                    G = nx.Graph()
                    G.add_edges_from(current_edges)
                    graphs.append(G)
                    current_edges = []
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0]); v = int(parts[1])
            except ValueError:
                u = parts[0]; v = parts[1]
            current_edges.append((u, v))
    if current_edges:
        G = nx.Graph()
        G.add_edges_from(current_edges)
        graphs.append(G)
    return graphs


# ---------------------------
# Preprocessing: run once per data graph
# ---------------------------
def preprocess_data_graph(datagraph_path: str) -> Dict[str, Any]:
    """
    执行对数据图的预处理（load -> bridge partition -> tutte -> block-cut -> index）。
    返回包含中间结果的字典，供后续每个 query 调用时使用。
    """
    result: Dict[str, Any] = {}
    t0 = time.time()
    data_graph = load_graph_from_txt(datagraph_path)
    t1 = time.time()
    result["data_graph"] = data_graph
    result["load_time"] = t1 - t0

    # Step 1: bridge decomposition (custom)
    ts = time.time()
    bridges, components = ppsm.custom_graph_partition(data_graph)
    te = time.time()
    result["bridges"] = bridges
    result["components"] = components
    result["custom_graph_partition_time"] = te - ts

    # Step 2: tutte (parallel)
    ts = time.time()
    subgraph_map, total_tutte = ppsm.tutte_from_bridge_decomposition(bridges, components)
    te = time.time()
    result["subgraph_map"] = subgraph_map
    result["total_tutte"] = total_tutte
    result["tutte_time"] = te - ts

    # Step 3: block-cut tree
    ts = time.time()
    block_cut_tree = ppsm.build_block_cut_tree(bridges, subgraph_map)
    te = time.time()
    result["block_cut_tree"] = block_cut_tree
    result["block_cut_tree_time"] = te - ts

    # Step 4: index
    ts = time.time()
    tutte_index = ppsm.build_rect_dominance_index_keep_self(subgraph_map)
    te = time.time()
    result["tutte_index"] = tutte_index
    result["tutte_index_time"] = te - ts

    result["preprocess_total_time"] = time.time() - t0
    return result


# ---------------------------
# Per-query processing with timeout
# ---------------------------
def process_single_query_with_timeout(
    query_graph: nx.Graph,
    k: int,
    preproc: Dict[str, Any],
    timeout_seconds: int
) -> Dict[str, Any]:
    """
    在预处理结果 preproc 上运行单个 query；设置超时（秒）。
    返回记录字典：包含 time_seconds, result_count, status, error(optional) 等。
    """
    # prepare return record
    rec: Dict[str, Any] = {
        "n_nodes": int(query_graph.number_of_nodes()),
        "n_edges": int(query_graph.number_of_edges()),
        "time_seconds": None,
        "candidate_size": None,
        #"filter_size": None,
        "search_count": None,
        "result_count": None,
        "status": None,
        "error": None
    }

    # prepare local references
    total_tutte = preproc["total_tutte"]
    subgraph_map = preproc["subgraph_map"]
    block_cut_tree = preproc["block_cut_tree"]
    tutte_index = preproc["tutte_index"]
    bridges = preproc["bridges"]

    # set timeout handler
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    start = time.time()
    try:
        # the actual per-query pipeline (same as your original logic)
        q_tutte = ppsm.approx_tutte_rectangle_maxonly(query_graph)
        if ppsm.maybe_contains_by_rect(q_tutte, total_tutte):
            candidate_set = ppsm.dispatch_query_processing(query_graph, subgraph_map, block_cut_tree, tutte_index, k, bridges)
            candidate_vertex_map, candidate_edge_set, candidate_path_dict = ppsm.construct_candidate_solution_space(query_graph, k, candidate_set, subgraph_map, bridges)
            rec["candidate_size"] = sum(len(v) for v in candidate_path_dict.values())
            f_candidate_vertex_map, f_candidate_edge_set, f_path_dict = ppsm.filter_by_global_weight_constraint(query_graph, k, candidate_vertex_map, candidate_edge_set, candidate_path_dict)
            #rec["filter_size"] = sum(len(pairs) for pairs in candidate_edge_set.values()) - sum(len(pairs) for pairs in f_candidate_edge_set.values())
            # run verification/backtracking to find matches
            matchs,search_count = ppsm.validate_and_find_matches(query_graph, k, f_candidate_vertex_map, f_path_dict)
            rec["search_count"] = search_count
            
        else:
            # trivially no candidates
            matchs = []

        end = time.time()
        rec["time_seconds"] = end - start
        
        # unique_subgraphs = {}
        # for g in matchs:
        #     key = frozenset(tuple(sorted(e)) for e in g.edges())
        #     if key not in unique_subgraphs:
        #         unique_subgraphs[key] = g
                    
        # rec["candidate_sugraphs"] = len(unique_subgraphs)
        # result count summary
        
        
        
        
        if isinstance(matchs, dict):
            rec["result_count"] = len(matchs)
        elif isinstance(matchs, (list, tuple, set)):
            rec["result_count"] = len(matchs)
        else:
            rec["result_count"] = None
        rec["status"] = "ok"
    except QueryTimeout:
        rec["time_seconds"] = timeout_seconds
        rec["result_count"] = None
        rec["status"] = "timeout"
        rec["error"] = {"type": "QueryTimeout", "msg": f"exceeded {timeout_seconds}s"}
    except Exception as e:
        rec["time_seconds"] = time.time() - start
        rec["result_count"] = None
        rec["status"] = "error"
        rec["error"] = {"type": type(e).__name__, "msg": str(e)}
    finally:
        signal.alarm(0)
        # restore previous handler
        signal.signal(signal.SIGALRM, old_handler)

    return rec


# ---------------------------
# Run queries in a single query-file and write JSON
# ---------------------------
def run_queries_and_record_single_file(
    querygraphs_path: str,
    k: int,
    datagraph_path: str,
    preproc: Dict[str, Any],
    timeout_seconds: int,
    out_json_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    在已经 preproc 的数据图上下文中，对 querygraphs_path 中的每个 query 运行并记录结果（带超时）。
    返回记录字典并写入 out_json_path（若提供或默认生成）。
    """
    # prepare output path
    if out_json_path is None:
        base_dir = os.path.dirname(querygraphs_path) or "."
        base_name = os.path.splitext(os.path.basename(querygraphs_path))[0]
        fname = f"{base_name}_k{k}_result.json"
        out_json_path = os.path.join(base_dir, fname)

    # load queries
    querygraphs = load_query_graphs_from_txt(querygraphs_path)

    per_query_records: List[Dict[str, Any]] = []
    total_query_start = time.time()
    for idx, query_graph in enumerate(querygraphs):
        print(f"[{os.path.basename(querygraphs_path)}] Query {idx}: n={query_graph.number_of_nodes()} e={query_graph.number_of_edges()} -> start")
        rec = process_single_query_with_timeout(query_graph, k, preproc, timeout_seconds)
        rec["index"] = idx
        print(f"[{os.path.basename(querygraphs_path)}] Query {idx} done: status={rec['status']} time={rec['time_seconds']:.3f}s matches={rec['result_count']}")
        per_query_records.append(rec)
    total_query_end = time.time()

    out: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data_graph": os.path.basename(datagraph_path),
        "query_graphs_file": os.path.basename(querygraphs_path),
        "k": k,
        "timeout": timeout_seconds,
        "preprocess": {
            "load_time": preproc.get("load_time"),
            "custom_graph_partition_time": preproc.get("custom_graph_partition_time"),
            "tutte_time": preproc.get("tutte_time"),
            "block_cut_tree_time": preproc.get("block_cut_tree_time"),
            "tutte_index_time": preproc.get("tutte_index_time"),
            "preprocess_total_time": preproc.get("preprocess_total_time")
        },
        "num_queries": len(querygraphs),
        "per_query": per_query_records
    }

    # ensure directory exists and write
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[{os.path.basename(querygraphs_path)}] All done. Total queries time: {total_query_end - total_query_start:.3f}s. Results saved to {out_json_path}")
    return out


# ---------------------------
# Batch runner over sets of query files and data files and k values
# ---------------------------
def batch_run(
    query_paths: List[str],
    data_paths: List[str],
    k_list: List[int],
    timeout_seconds: int,
    output_root: str = "./results"
):
    """
    对 query_paths 中的每个查询文件、data_paths 中的每个数据图、每个 k 做批量实验。
    输出路径自动构造为: ./results/k{k}_t{timeout}/{data_basename}/{query_basename}_k{k}.json
    """
    os.makedirs(output_root, exist_ok=True)

    for data_path in data_paths:
        # preprocess once per data graph
        print("============================================================")
        print("Preprocessing data graph:", data_path)
        preproc = preprocess_data_graph(data_path)

        data_basename = os.path.splitext(os.path.basename(data_path))[0]

        for k in k_list:
            for qpath in query_paths:
                q_basename = os.path.splitext(os.path.basename(qpath))[0]
                out_dir = os.path.join(output_root, f"k{k}_t{timeout_seconds}", data_basename)
                os.makedirs(out_dir, exist_ok=True)
                out_json = os.path.join(out_dir, f"{q_basename}_k{k}_ppsm.json")

                # if already exists, skip (resume capability)
                # if os.path.exists(out_json):
                #     print(f"Skip existing: {out_json}")
                #     continue

                print(f"Running: data={data_basename}, query={q_basename}, k={k}")
                try:
                    run_queries_and_record_single_file(qpath, k, data_path, preproc, timeout_seconds, out_json_path=out_json)
                except Exception as e:
                    print(f"ERROR running {qpath} on {data_path} (k={k}): {type(e).__name__}: {e}")


# ---------------------------
# Helper: collect query files under directory (all .txt)
# ---------------------------
def collect_query_files(query_dir_or_list) -> List[str]:
    if isinstance(query_dir_or_list, str):
        if os.path.isdir(query_dir_or_list):
            files = sorted(glob.glob(os.path.join(query_dir_or_list, "*.txt")))
            return files
        elif os.path.isfile(query_dir_or_list):
            return [query_dir_or_list]
        else:
            return []
    elif isinstance(query_dir_or_list, (list, tuple)):
        out = []
        for p in query_dir_or_list:
            out.extend(collect_query_files(p))
        return out
    else:
        return []


# ---------------------------
# Example main (edit parameters here)
# ---------------------------
if __name__ == "__main__":
    # ---- edit these variables as needed ----
    #DATA_FILES = ["./data/ca-CSphd.mtx"]
    DATA_FILES = ["./bio-CE-HT.txt"]
    K_LIST = [1,2,3,4]                  # list of k values to test
    TIMEOUT_SECONDS = 600            # per-query timeout (seconds)
    OUTPUT_ROOT = "./results"        # where to save results
    # ---------------------------------------

    query_files=["./queries/all_connected_E5.txt"]
    #query_files=["./queries/multicycle_E8.txt","./queries/unicyclic_E8.txt","./queries/tree_E8.txt"]
    print("Collected query files:", query_files)

    batch_run(
        query_paths=query_files,
        data_paths=DATA_FILES,
        k_list=K_LIST,
        timeout_seconds=TIMEOUT_SECONDS,
        output_root=OUTPUT_ROOT
    )