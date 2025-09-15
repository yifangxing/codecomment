import argparse
import csv
import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict
import subprocess
import sys
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
import numpy as np
import pandas as pd
from tqdm import tqdm
from test_util import reverse_topological_sort, read_lines_from_file,get_comment, save_nodes_to_csv,save_node_to_csv,clean_java_comment,generate_comment

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
   
def count_nodes_per_level(result):
    """
    统计每个层级的节点总数。
    """
    level_counts = Counter()

    for _, level in result:
        level_counts[level] += 1

    return dict(level_counts)

def divide_levels(level_counts):
    """
    将层级划分为4个部分:
    - 第1层不考虑
    - 第2层作为第1部分
    - 剩余层级划分为3部分，尽量保证节点数均衡，且每个层级必须整体落在某个部分
    """
    levels = sorted(level_counts.keys())
    
    # 忽略第1层
    levels = [lvl for lvl in levels if lvl != 1]
    if not levels:
        return []

    # 第2层单独拿出来
    part1 = (2, 2)
    remaining_levels = [lvl for lvl in levels if lvl > 2]

    # 如果没有更多层级，直接返回
    if not remaining_levels:
        return [part1]

    # 剩余层级总数
    total_nodes = sum(level_counts[lvl] for lvl in remaining_levels)
    target = total_nodes / 3  # 每部分目标节点数

    parts = []
    current_sum = 0
    start = remaining_levels[0]
    count_parts = 0

    for i, lvl in enumerate(remaining_levels):
        current_sum += level_counts[lvl]
        # 如果达到或超过目标，并且还能保证后面能分完
        if (current_sum >= target and count_parts < 2) or (i == len(remaining_levels)-1):
            end = lvl
            parts.append((start, end))
            count_parts += 1
            if i < len(remaining_levels)-1:
                start = remaining_levels[i+1]
            current_sum = 0

    return [part1] + parts

def compute_score(node, nodes_info, in_degree, dataflow, count, child_node, mapping, output_file):
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
        count_score = round(count_score, 2)  # 保留两位小数
        dataflow_score = round(dataflow_score, 2)
        indegree_score = round(indegree_score, 2)
        total_score = round(total_score, 2)  
        # # 替换为掩码名称
        # if method_name in mapping:
        #     method_name = mapping[method_name]
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

def mask_java_code(method_code: str):
    """
    使用 CodeMasker 将 Java 代码中的方法名称掩盖。
    """
    # 调用 java -jar
    result = subprocess.run(
        ["/usr/lib/jvm/java-17-openjdk-amd64/bin/java", "-cp", "process-1.0-SNAPSHOT-jar-with-dependencies.jar", "CodeMasker", method_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    code_part, map_part = result.stdout.split("===MAP===")
    code = code_part.split("===CODE===")[1].strip()
    mapping = {}
    for line in map_part.strip().splitlines():
        if "->" in line:
            orig, masked = line.split("->")
            mapping[orig.strip()] = masked.strip()

    return code, mapping

def main(config):
    dot_file = config["dot_file"] # 存储调用图地址
    src_path = config["src_path"] # 项目地址
    csv_file_path = config["csv_file_path"] # 存储第二次数据地址
    comment_file_path = config["comment_file_path"] # 存储原方法注释地址
    order_file_path = config["order_file_path"] # 存储子方法出现顺序地址
    dataflow_file_path = config["dataflow_file_path"] # 存储数据流信息地址
    counts_file_path = config["counts_file_path"] # 存储子方法出现次数地址
    class_doc_file_path = config["class_doc_file_path"] # 存储类功能描述地址
    score_file_path = config["score_file_path"] # 存储第二次方法得分地址
    cycle_info_file_path = config["cycle_info_file_path"] # 存储方法执行顺序地址
    model = config["model"] # 使用的大模型
    # 方法调用的顺序,外部方法数量，孩子节点,每个节点的入度
    cycle_info, lenght, child_nodes, in_degree = reverse_topological_sort(dot_file)
    # print(cycle_info)
    pre_comments = get_precomment(comment_file_path)
    orders = get_order(order_file_path)
    dataflows = get_order(dataflow_file_path)
    counts = get_counts(counts_file_path)
    nodes_info = parse_dot_graph(dot_file) # 存储从调用图获取的节点信息
    nodes = {}  # 存储所有节点信息
    class_doc = {}  # 存储类的功能描述
    with open(class_doc_file_path, "r", encoding="utf-8") as f:
        class_doc = json.load(f)  # 读取类的功能描述
    with open(cycle_info_file_path, "w", encoding="utf-8") as f:
        json.dump(cycle_info, f, ensure_ascii=False, indent=4) # 保存调用顺序信息
    # with open(cycle_info_file_path, "r", encoding="utf-8") as f:
    #     cycle_info = json.load(f)
    level_counts = count_nodes_per_level(cycle_info) # 统计每个层级节点数量
    parts = divide_levels(level_counts) # 将层级划分为4部分
    # 如果循环执行到一半中断，可以从上次中断的 i 继续，读取 csv 文件中已经有前 i-1 个节点的信息，并读取处理顺序（保证和中断前一致）
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
    #     node_level = row['node_level']
    #     nodes[full_name] = Node(file_path, method_name,full_name,Start_Line, End_Line, child_name, None, comment, pre_comment, bool, node_level)
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
        
    start_i = 0  # 假设从第0个节点开始
    for i, result in enumerate(tqdm(cycle_info), start=1):
        if i <= start_i:
            continue  # 跳过已完成的部分
        node = result[0] 
        node_level = result[1] 
        # if node_level > max:
        #     max = node_level      
        path = nodes_info[node]["path"]
        prefix, method_sig = node.split(':')
        method_name = method_sig.split('(')[0]
        class_label = node.split(":")[0]
        nodes[node] = None
        # 外部库函数
        if i<=lenght:
            # 判断是否外部联网检索该函数注释(未来实现)
            if path:
                # 节点所属的文件地址
                file_path = src_path + path + ".java"
                os.path.join(file_path)
                if os.path.exists(file_path):
                    # print("file_path:",file_path,"node:",node)
                    pre_comment = pre_comments.get(node, "")
                    java_code = read_lines_from_file(file_path, nodes_info[node]["start_line"], nodes_info[node]["end_line"])
                    # if method_name == "<init>":
                    #     comment = get_comment(node, nodes, method_name, None, java_code, None, None, class_doc[class_label], model)
                    # else:
                    #     masked_code, mapping = mask_java_code(java_code)
                    #     comment = get_comment(node, nodes, method_name, None, masked_code, None, None, class_doc[class_label], model)
                    comment = get_comment(node, nodes, method_name, None, java_code, None, None, class_doc[class_label], model)
                    nodes[node] = Node(file_path, method_name,node, nodes_info[node]["start_line"], nodes_info[node]["end_line"], None, class_doc[class_label], comment, pre_comment, False, node_level)
            else:
                comment = get_comment(node, nodes, method_name, None, None, None, None, None, model)
                nodes[node] = Node(None, method_name,node, None, None, None, None, comment, "", False, node_level)
        else:
            if path:
                # print("内部函数")
                file_path = src_path + path + ".java"
                os.path.join(file_path)
                if os.path.exists(file_path):
                    pre_comment = pre_comments.get(node, "")
                    order = orders.get(node, {})
                    dataflow = dataflows.get(node, [])
                    count = counts.get(node, {})
                    child_node = child_nodes[node] #子方法集合
                    java_code = read_lines_from_file(file_path, nodes_info[node]["start_line"], nodes_info[node]["end_line"]) 
                    comment = ""              
                    # if method_name == "<init>":
                    #     score = compute_score(node,nodes_info, in_degree, dataflow, count, child_node, {}, score_file_path)
                    #     comment = get_comment(node, nodes, method_name, child_node, java_code, order, score, class_doc[class_label],model)
                    # else:
                    #     masked_code, mapping = mask_java_code(java_code)
                    #     score = compute_score(node,nodes_info, in_degree, dataflow, count, child_node, mapping, score_file_path)
                    #     comment = generate_comment(node, nodes, method_name, child_node, masked_code, order, score, class_doc[class_label], mapping, model)
                    score = compute_score(node,nodes_info, in_degree, dataflow, count, child_node, {}, score_file_path)
                    comment = get_comment(node, nodes, method_name, child_node, java_code, order, score, class_doc[class_label],model)
                    nodes[node] = Node(file_path, method_name,node, nodes_info[node]["start_line"], nodes_info[node]["end_line"], child_node, class_doc[class_label], comment, pre_comment, True, node_level)
            
                    # if comment !="" and pre_comment != "" and pre_comment :
                    #     def multiline_to_single_line(text: str) -> str:
                    #         """
                    #         将多行字符串转换为单行.
                    #         """
                    #         return re.sub(r'\s+', ' ', text).strip()
                    #     def write_to_dot(file_path, content):
                    #         """
                    #         将内容写入 DOT 文件，并在行首添加行号。
                            
                    #         :param file_path: DOT 文件路径
                    #         :param content: 要写入的内容
                    #         :return: 返回写入的行号
                    #         """
                    #         try:
                    #             # 首先读取当前文件的行数来确定新行号
                    #             with open(file_path, 'r', encoding='utf-8') as file:
                    #                 line_number = sum(1 for _ in file) + 1  # 计算当前行数+1
                                
                    #             # 以追加模式写入带行号的内容
                    #             with open(file_path, 'a', encoding='utf-8') as file:
                    #                 line_content = f"{line_number} {content}\n"  # 格式: "行号 内容"
                    #                 file.write(line_content)
                                
                    #             return line_number  # 返回写入的行号
                            
                    #         except FileNotFoundError:
                    #             # 如果文件不存在，从行号1开始
                    #             with open(file_path, 'w', encoding='utf-8') as file:
                    #                 line_content = f"1 {content}\n"  # 第一行
                    #                 file.write(line_content)
                    #             return 1
                            
                    #     pre_comment = clean_java_comment(pre_comment)
                    #     pre_comment = multiline_to_single_line(pre_comment)
                    #     comment = clean_java_comment(comment)
                    #     comment = multiline_to_single_line(comment)
                    #     start_levels, end_levels = parts[1]
                    #     start_levels1, end_levels1 = parts[2]
                    #     if node_level == 2 :
                    #         level = "1"
                    #     elif node_level >= start_levels and node_level <= end_levels:
                    #         level = "2"
                    #     elif node_level >= start_levels1 and node_level <= end_levels1:
                    #         level = "3"
                    #     else:
                    #         level = "4"
                    #     write_to_dot("/home/yfx/codecomment/codecomment/result/java/pdfbox/level/"+level+"/hyp.comment", comment) #大模型生成的注释
                    #     write_to_dot("/home/yfx/codecomment/codecomment/result/java/pdfbox/level/"+level+"/ref.comment", pre_comment) #参考注释

        save_node_to_csv(nodes[node], csv_file_path) 

    

    # save_nodes_to_csv(nodes, csv_file_path)

# 示例调用
if __name__ == "__main__":
    config_str = sys.argv[1]
    config = json.loads(config_str)
    main(config)
    
    