import networkx as nx
import time
import json
import os
import signal

from naive import count_homeomorphic_subgraphs


# =====================================
# 超时异常
# =====================================

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


import os


def load_all_txt_files(dir_path):
    """
    加载目录下所有 .txt 文件（绝对路径）
    """
    files = []

    for name in os.listdir(dir_path):

        if name.endswith(".txt"):

            full = os.path.join(dir_path, name)

            if os.path.isfile(full):
                files.append(full)

    files.sort()

    return files

# =====================================
# 读数据图
# =====================================

def load_single_graph(path):

    G = nx.Graph()

    with open(path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            u, v = line.split()
            G.add_edge(int(u), int(v))

    return G


# =====================================
# 读查询图（空行分隔）
# =====================================

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


# =====================================
# 文件名前缀
# =====================================

def get_prefix(path):

    name = os.path.basename(path)

    return os.path.splitext(name)[0]


# =====================================
# 自动构造输出目录
# =====================================

def build_output_dir(query_file, data_file, k, timeout):

    data_prefix = get_prefix(data_file)

    kt_dir = f"k{k}_t{timeout}"

    base = "results"

    return os.path.join(base, kt_dir, data_prefix)


# =====================================
# 单查询（带超时）
# =====================================

def run_single_query(G_data, G_query, k, timeout):

    subgraphs = 0
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    start = time.time()

    try:

        cnt, _ ,subgraphs= count_homeomorphic_subgraphs(
            G_data,
            G_query,
            k,
            verbose=False
        )

        elapsed = time.time() - start

        signal.alarm(0)

        return "ok", cnt, elapsed, subgraphs


    except TimeoutException:

        return "timeout", None, timeout, subgraphs


    except Exception as e:

        return "error", None, str(e), subgraphs



# =====================================
# 批处理主函数
# =====================================

def batch_run(
    query_file: str,
    data_file: str,
    k: int,
    timeout: int = 60
):

    # ---------- 自动输出目录 ----------

    output_dir = build_output_dir(
        query_file, data_file, k, timeout
    )

    os.makedirs(output_dir, exist_ok=True)


    print("Output dir:", output_dir)

    # ---------- 读图 ----------

    print("Loading data graph...")
    G_data = load_single_graph(data_file)

    print("Loading query graphs...")
    queries = load_query_graphs(query_file)

    print(f"Loaded {len(queries)} queries")


    results = {
        "query_file": query_file,
        "data_file": data_file,
        "k": k,
        "timeout": timeout,
        "num_queries": len(queries),
        "results": []
    }


    # ---------- 跑查询 ----------

    for i, Q in enumerate(queries):

        print(f"Running Query {i} "
              f"(V={Q.number_of_nodes()}, "
              f"E={Q.number_of_edges()})")

        status, cnt, info, subgraphs = run_single_query(
            G_data, Q, k, timeout
        )

        rec = {
            "query_id": i,
            "num_nodes": Q.number_of_nodes(),
            "num_edges": Q.number_of_edges(),
            "candidate_subgraphs": subgraphs,
            "status": status
        }

        if status == "ok":

            rec["time"] = info
            rec["match_count"] = cnt

        elif status == "timeout":

            rec["time"] = timeout
            rec["match_count"] = None

        else:

            rec["time"] = None
            rec["match_count"] = None
            rec["error"] = info


        results["results"].append(rec)

        print(" ->", status)


    # ---------- 文件名 ----------

    q = get_prefix(query_file)

    fname = f"{q}_naive.json"

    out_path = os.path.join(output_dir, fname)


    # ---------- 写 JSON ----------

    with open(out_path, "w", encoding="utf-8") as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )


    print("=================================")
    print("Saved:", out_path)
    print("=================================")


    return out_path


def run_all_experiments(
    query_dir,
    data_dir,
    k_list,
    timeout
):

    # 加载全部文件
    query_files = load_all_txt_files(query_dir)
    data_files = load_all_txt_files(data_dir)

    print("Loaded Queries:", len(query_files))
    print("Loaded Graphs :", len(data_files))

    # 三重循环
    for qf in query_files:

        for df in data_files:

            for k in k_list:

                print("=" * 60)
                print("Query:", qf)
                print("Data :", df)
                print("k    :", k)

                try:

                    batch_run(
                        query_file=qf,
                        data_file=df,
                        k=k,
                        timeout=timeout
                    )

                except Exception as e:

                    print("ERROR:", e)

# =====================================
# 主入口
# =====================================

if __name__ == "__main__":

    #QUERY_DIR = 

    DATA_FILES = [
        "./graphs/ws_Et20_Er70_n35.txt","./graphs/ba_Et20_Er34_n35.txt","./graphs/block_Et20_Er21_n10.txt","./graphs/er_Et20_Er21_n44.txt"
    ]

    K_LIST = [1]

    TIMEOUT = 600


    #query_files = load_all_txt_files(QUERY_DIR)
    query_files=["./queries/all_connected_E3.txt","./queries/all_connected_E4.txt","./queries/all_connected_E5.txt"]

    print("Loaded queries:", len(query_files))
    print("Using graphs :", DATA_FILES)
    print("Using k list :", K_LIST)


    for qf in query_files:

        for df in DATA_FILES:

            for k in K_LIST:

                print("=" * 60)
                print("Query:", qf)
                print("Data :", df)
                print("k    :", k)

                batch_run(
                    query_file=qf,
                    data_file=df,
                    k=k,
                    timeout=TIMEOUT
                )