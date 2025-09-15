import csv
import json
import os
import re
from pathlib import Path
from collections import defaultdict
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
import numpy as np
import pandas as pd
from tqdm import tqdm
from util import reverse_topological_sort, read_lines_from_file,get_node_comment, save_nodes_to_csv,save_node_to_csv, get_class_comment


class Node:
    def __init__(self, path, name, full_name, start_line, end_line, child_name, domain, comment, pre_comment, is_inner_method, node_level ):
        self.path = path # 节点所属的文件地址
        self.name = name # 节点的方法名
        self.full_name = full_name # 节点的方法全名
        self.start_line = start_line # 方法开始行号
        self.end_line = end_line # 方法结束行号
        self.child_name = child_name  # 节点的子方法列表
        self.domain = domain # 节点的所属类的功能描述
        self.comment = comment # 大模型生成的注释
        self.pre_comment = pre_comment # 该方法的原注释
        self.is_inner_method = is_inner_method # 是否为项目中的方法
        self.node_level = node_level  # 节点在调用图中所属层级
    def __repr__(self):
        return (f"Node(path={self.path}, name={self.name}, full_name={self.full_name}, start_line={self.start_line}, end_line={self.end_line}, "
                f"child_name={self.child_name}, domain={self.domain}, comment={self.comment}, pre_comment={self.pre_comment},is_inner_method={self.is_inner_method },node_level={self.node_level })")
def parse_dot_graph(dot_path):
    """ 解析 DOT 文件，提取调用图的节点信息。"""
    G = read_dot(dot_path)
    nodes_info = {}

    for node_id, attrs in G.nodes(data=True):
        label = attrs.get("label", "").strip('"')

        idx = label.find(")")
        method_sig = label[:idx + 1]
        path_line = label[idx + 1:]

        # 提取路径（去除 .class 后缀，去除 $ 内部类）
        path_line_match = re.match(r"(.*?)(?:\$.*)?\.class(?::(\d+)-(\d+))?", path_line)

        if path_line_match:
            raw_path = path_line_match.group(1)
            start_line = int(path_line_match.group(2)) if path_line_match.group(2) else None
            end_line = int(path_line_match.group(3)) if path_line_match.group(3) else None

            # 转成以 / 开头的路径
            file_path = "/" + raw_path if not raw_path.startswith("/") else raw_path
        else:
            file_path = None
            start_line = None
            end_line = None

        # 存储每个节点信息
        nodes_info[node_id] = {
            "method": method_sig,
            "path": file_path,
            "start_line": start_line,
            "end_line": end_line
        }
        #print(f"Processing node {node_id}, method_sig:{method_sig}, file_path:{file_path}, start_line:{start_line}, end_line:{end_line}")
    return nodes_info


def get_precomment(path):
    """获取原方法的注释。"""

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_order(path):
    """获取子方法在源代码中的调用顺序。"""

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_dataflow(path):
    """获取子方法之间的数据依赖关系。"""

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def get_counts(path):
    """获取子方法在源代码中的调用次数。"""

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)   
    return data

def process(file_path, nodes, node, nodes_info, child_node, score, order, method_name, path, class_list):
    """返回大模型生成的注释和孩子节点注释是否存在"""
    node_info = nodes_info[node]
    # file_path_no_ext = path.split('.')[0]
    comment = ""
    if node_info["start_line"] is None or node_info["end_line"] is None:
        comment,class_list = get_node_comment(node, nodes, method_name, None, None, None, None, path, class_list)
        return comment,class_list
    else:
        code = read_lines_from_file(file_path, node_info["start_line"], node_info["end_line"]) #获取方法代码
        # 生成注释
        comment,class_list = get_node_comment(node, nodes, method_name, child_node, code, order, score, path, class_list)
        return comment,class_list

def process_leaf(file_path, nodes, node, method_name, nodes_info, class_list):
    """返回大模型生成的注释和孩子节点注释是否存在"""
    node_info = nodes_info[node]
    comment = ""
    code = read_lines_from_file(file_path, node_info["start_line"], node_info["end_line"]) #获取方法代码
    # 生成注释
    comment,class_list = get_node_comment(node, nodes, method_name, None, code, None, None, None, class_list)

    return comment,class_list
   

def compute_score(node, nodes_info, in_degree, dataflow, count, child_node, output_file):
    """计算每个子方法的重要性得分，并追加保存到 JSON 文件。"""
    node_info = nodes_info[node]
    if node_info["start_line"] is None or node_info["end_line"] is None:
        return {}

    # 初始化存储
    child_in_degree, child_dataflow, child_counts = {}, {}, {}

    # 统计入度、出现次数
    count_tal = 0
    for child in child_node:
        prefix, method_sig = child.split(':')
        method_name = method_sig.split('(')[0]
        child_dataflow.setdefault(method_name, 0)
        degree = in_degree.get(child, 0)
        child_in_degree[method_name] = degree
        child_counts[method_name] = count.get(method_name, 0)
        count_tal += count.get(method_name, 0)

    # 统计数据流枢纽度
    for item in dataflow:
        if item["sourceMethod"] in child_dataflow:
            child_dataflow[item["sourceMethod"]] += 1
        if item["targetMethod"] in child_dataflow:
            child_dataflow[item["targetMethod"]] += 1

    # 归一化时避免除零
    def normalize(val, min_v, max_v):
        return 0.0 if max_v == min_v else (val - min_v) / (max_v - min_v)

    max_in_degree, min_in_degree = max(child_in_degree.values(), default=0), min(child_in_degree.values(), default=0)
    max_dataflow, min_dataflow = max(child_dataflow.values(), default=0), min(child_dataflow.values(), default=0)

    scores,score = {},{}
    for child in child_node:
        prefix, method_sig = child.split(':')
        method_name = method_sig.split('(')[0]

        count_score = child_counts[method_name] / count_tal if count_tal > 0 else 0
        dataflow_score = normalize(child_dataflow[method_name], min_dataflow, max_dataflow)
        # 对数缩放
        log_values = np.log(child_in_degree[method_name]) / np.log(2)  # 换底公式
        log_max_in_degree = np.log(max_in_degree) / np.log(2)
        log_min_in_degree = np.log(min_in_degree) / np.log(2)
        indegree_score = normalize(log_values, log_min_in_degree, log_max_in_degree)

        total_score = 0.3 * count_score + 0.3 * dataflow_score + 0.4 * indegree_score

        scores[method_name] = {
            "count_score": count_score,
            "dataflow_score": dataflow_score,
            "indegree_score": indegree_score,
            "total_score": total_score
        }
        score[method_name] = total_score

    # === 追加写入 JSON 文件 ===
    record = {node: scores}

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.append(record)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    return score


# 示例调用
if __name__ == "__main__":
    print("开始处理...")
    
    dot_file = r"/home/yfx/codecomment/codecomment/methodline/java/new/hadoop-common/updated.dot" # 存储调用图地址
    src_path = r"/home/yfx/codecomment/codecomment/project/java" # 项目地址
    csv_file_path = r"/home/yfx/codecomment/codecomment/result/java/hadoop-common/class/lessdata.csv" # 存储结果数据地址
    comment_file_path = r"/home/yfx/codecomment/codecomment/methodline/java/new/hadoop-common/method_comments.json" # 存储原方法注释地址
    order_file_path = r"/home/yfx/codecomment/codecomment/methodline/java/new/hadoop-common/order.json" # 存储子方法出现顺序地址
    dataflow_file_path = r"/home/yfx/codecomment/codecomment/methodline/java/new/hadoop-common/dataflow.json" # 存储数据流信息地址
    counts_file_path = r"/home/yfx/codecomment/codecomment/methodline/java/new/hadoop-common/counts.json" # 存储子方法出现次数地址
    class_doc_file_path = r"/home/yfx/codecomment/codecomment/result/java/hadoop-common/class/class_doc.json" # 存储类功能描述地址
    class_list_path = r"/home/yfx/codecomment/codecomment/result/java/hadoop-common/class/class_list.json" # 存储类和其子方法注释地址
    cycle_info_file_path = r"/home/yfx/codecomment/codecomment/result/java/hadoop-common/class/cycle_info11.json" # 存储方法执行顺序地址
    score_file_path = r"/home/yfx/codecomment/codecomment/result/java/hadoop-common/class/score111.json" # 存储方法得分地址
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(["file_path", "Name","full_name", "Start Line", "End Line", "Comment", "Pre_Comment", "child Name", "domain", "inner_method", "node_level"])
    else:
        # 检查文件是否为空
        if os.path.getsize(csv_file_path) == 0:
            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # 写入表头
                writer.writerow(["file_path",  "Name","full_name", "Start Line", "End Line", "Comment", "Pre_Comment", "child Name", "domain", "inner_method", "node_level"])
        else:
            # print("文件不为空，跳过写入表头")
            pass
    # 方法调用的顺序,外部方法数量，孩子节点,每个节点的入度
    cycle_info , lenght ,child_nodes,in_degree = reverse_topological_sort(dot_file)
    with open(cycle_info_file_path, "w", encoding="utf-8") as f:
            json.dump(cycle_info, f, ensure_ascii=False, indent=4)
    # with open(cycle_info_file_path, "r", encoding="utf-8") as f:
    #     cycle_info = json.load(f)
    pre_comments = get_precomment(comment_file_path)
    orders = get_order(order_file_path)
    dataflows = get_order(dataflow_file_path)
    counts = get_counts(counts_file_path)
    nodes_info = parse_dot_graph(dot_file)
    nodes = {}
    class_list = {} # 记录项目中的类

    # # 如果循环执行到一半中断，可以从上次中断的 i 继续
    # start_line = 0  # 假设从第0行开始读取
    # df = pd.read_csv(csv_file_path)
    # for idx, row in tqdm(df.iloc[start_line:].iterrows(), total=len(df)-start_line):
    #     file_path = str(row['file_path'])
    #     method_name = str(row['Name'])
    #     full_name = str(row['full_name'])
    #     bool = row['inner_method']
    #     Start_Line = row['Start Line']
    #     End_Line = row['End Line']
    #     comment = str(row['Comment'])
    #     pre_comment = str(row['Pre_Comment'])
    #     child_name = str(row['child Name'])
    #     nodes[full_name] = Node(file_path, method_name,full_name,Start_Line, End_Line, child_name, None, comment, pre_comment, bool)
    # with open(class_list_path, "r", encoding="utf-8") as f:
    #     class_list = json.load(f)
    start_i = 0  # 假设从第0个节点开始，即csv中行号减1开始
    for i, result in enumerate(tqdm(cycle_info), start=1):
        if i <= start_i:
            continue  # 跳过已完成的部分 
        node = result[0] 
        node_level = result[1] 
        path = nodes_info[node]["path"]
        nodes[node] = None
        prefix, method_sig = node.split(':')
        method_name = method_sig.split('(')[0]
        # 外部库函数
        if i<=lenght:
            # 判断是否外部联网检索该函数注释(未来实现)
            if path:
                # 节点所属的文件地址
                file_path = src_path + path + ".java"
                os.path.join(file_path)
                if os.path.exists(file_path):
                    # print("file_path:",file_path,"node:",node)
                    pre_comment = pre_comments.get(node, {})
                    comment,class_list = process_leaf(file_path, nodes, node, method_name, nodes_info, class_list)
                    nodes[node] = Node(file_path, method_name,node, nodes_info[node]["start_line"], nodes_info[node]["end_line"], None, None, comment, pre_comment, False,node_level)
            else:
                comment,class_list = process(None, nodes, node, nodes_info, None, None, None, method_name, None, class_list)
                nodes[node] = Node(None, method_name,node, None, None, None, None, comment, "", False,node_level)

        else:
            if path:
                # print("内部函数")
                file_path = src_path + path + ".java"
                os.path.join(file_path)
                if os.path.exists(file_path):
                    pre_comment = pre_comments.get(node, {})
                    order = orders.get(node, {})
                    dataflow = dataflows.get(node, {})
                    count = counts.get(node, {})
                    child_node = child_nodes[node] #子方法集合
                    score = compute_score(node,nodes_info, in_degree, dataflow, count, child_node,score_file_path)
                    prefix, method_sig = node.split(':')
                    method_name = method_sig.split('(')[0]
                    comment,class_list = process(file_path, nodes, node, nodes_info, child_node, score, order, method_name, path, class_list)
                    nodes[node] = Node(file_path, method_name,node, nodes_info[node]["start_line"], nodes_info[node]["end_line"], child_nodes[node], None, comment, pre_comment, True,node_level)
            else:
                nodes[node] = Node(None, method_name,node, None, None, child_nodes[node], None, "", "", True,node_level)
        save_node_to_csv(nodes[node], csv_file_path)

        with open(class_list_path, "w", encoding="utf-8") as f:
            json.dump(class_list, f, ensure_ascii=False, indent=4)
 
    # 生成类的功能描述
    class_doc = {}  # 存储类的功能描述
    for class_node,methods_descriptions in tqdm(class_list.items()):
        doc = get_class_comment(class_node,methods_descriptions)
        class_doc[class_node] = doc
    with open(class_doc_file_path, "w", encoding="utf-8") as f:
            json.dump(class_doc, f, ensure_ascii=False, indent=4)