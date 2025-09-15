
import csv
import json
import os
import re
from collections import Counter
import sys
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
from tqdm import tqdm
from util import reverse_topological_sort, read_lines_from_file,get_comment, save_node_to_csv

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

def main(config):
    dot_file = config["dot_file"] # 存储调用图地址
    src_path = config["src_path"] # 项目地址
    csv_file_path = config["csv_file_path"] # 存储第二次数据地址
    comment_file_path = config["comment_file_path"] # 存储原方法注释地址
    order_file_path = config["order_file_path"] # 存储子方法出现顺序地址
    class_doc_file_path = config["class_doc_file_path"] # 存储类功能描述地址
    cycle_info_file_path = config["cycle_info_file_path"] # 存储方法执行顺序地址
    model = config["model"] # 使用的大模型
    # 方法调用的顺序,外部方法数量，孩子节点,每个节点的入度
    cycle_info, lenght, child_nodes, in_degree = reverse_topological_sort(dot_file)
    # print(cycle_info)
    pre_comments = get_precomment(comment_file_path)
    orders = get_order(order_file_path)
    nodes_info = parse_dot_graph(dot_file) # 存储从调用图获取的节点信息
    nodes = {}  # 存储所有节点信息
    class_doc = {}  # 存储类的功能描述
    with open(class_doc_file_path, "r", encoding="utf-8") as f:
        class_doc = json.load(f)  # 读取类的功能描述
    with open(cycle_info_file_path, "w", encoding="utf-8") as f:
        json.dump(cycle_info, f, ensure_ascii=False, indent=4) # 保存调用顺序信息
    # with open(cycle_info_file_path, "r", encoding="utf-8") as f:
    #     cycle_info = json.load(f)
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
                    pre_comment = pre_comments.get(node, "")
                    java_code = read_lines_from_file(file_path, nodes_info[node]["start_line"], nodes_info[node]["end_line"])
                    comment = get_comment(node, nodes, method_name, None, java_code, None, class_doc[class_label], model)
                    nodes[node] = Node(file_path, method_name,node, nodes_info[node]["start_line"], nodes_info[node]["end_line"], None, class_doc[class_label], comment, pre_comment, False, node_level)
            else:
                comment = get_comment(node, nodes, method_name, None, None, None, None, model)
                nodes[node] = Node(None, method_name,node, None, None, None, None, comment, "", False, node_level)
        else:
            if path:
                # print("内部函数")
                file_path = src_path + path + ".java"
                os.path.join(file_path)
                if os.path.exists(file_path):
                    pre_comment = pre_comments.get(node, "")
                    order = orders.get(node, {})
                    child_node = child_nodes[node] #子方法集合
                    java_code = read_lines_from_file(file_path, nodes_info[node]["start_line"], nodes_info[node]["end_line"]) 
                    comment = ""              
                    comment = get_comment(node, nodes, method_name, child_node, java_code, order, class_doc[class_label],model)
                    nodes[node] = Node(file_path, method_name,node, nodes_info[node]["start_line"], nodes_info[node]["end_line"], child_node, class_doc[class_label], comment, pre_comment, True, node_level)
            

        save_node_to_csv(nodes[node], csv_file_path) 

if __name__ == "__main__":
    config_str = sys.argv[1]
    config = json.loads(config_str)
    main(config)
    
    