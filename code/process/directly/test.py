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
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

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

def reverse_topological_sort(dot_file_path):
    """
    返回:
        - 节点处理顺序及其层级 (list of tuples: (node, level))
        - 叶子节点个数
        - 每个节点的孩子节点字典
        - 每个节点的入度字典
    """
    # 读取 DOT 文件构建图
    G = nx.nx_pydot.read_dot(dot_file_path)

    # 计算每个节点的出度
    out_degree = dict(G.out_degree())
  
    # 存储所有出度为0的节点（叶子节点）
    zero_out_degree_queue = [node for node, degree in out_degree.items() if degree == 0]
    length = len(zero_out_degree_queue)

    result = []   # 存储 (node, level)
    level_map = {}  # 存储节点对应的层级

    # 初始化叶子节点层级为 1
    for node in zero_out_degree_queue:
        level_map[node] = 1

    # BFS 遍历，层级从叶子往上传播
    while zero_out_degree_queue:
        u = zero_out_degree_queue.pop(0)
        result.append((u, level_map[u]))

        for v in G.predecessors(u):
            out_degree[v] -= 1
            # 父节点层级 = max(已有值, 当前子节点层级 + 1)
            level_map[v] = max(level_map.get(v, 0), level_map[u] + 1)
            if out_degree[v] == 0:
                zero_out_degree_queue.append(v)

    # 检查是否存在环
    if len(result) != len(G):
        for v in out_degree:
            if out_degree[v] > 0:
                print(f"节点 {v} 的出度为 {out_degree[v]}，可能存在环")
        raise ValueError("Graph contains a cycle")

    return result, length

def get_precomment(path):
    """获取原方法的注释。"""

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def read_lines_from_file(file_path, start_line, end_line):
    """
    从文件中读取指定行范围的内容。
    :param file_path: 文件路径
    :param start_line: 起始行号（包含）
    :param end_line: 结束行号（包含）
    :return: 指定行范围的内容字符串
    """
    lines = []
    with open(file_path, "r") as file:
        for current_line, line_content in enumerate(file, start=1):
            if start_line <= current_line <= end_line:
                lines.append(line_content)
            elif current_line > end_line:
                break
    # 将列表中的行内容拼接成一个字符串
    return ''.join(lines)

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

def process(file_path, node, nodes_info, method_name, model):
    """返回大模型生成的注释."""
    node_info = nodes_info[node]
    comment = ""
    code = read_lines_from_file(file_path, node_info["start_line"], node_info["end_line"]) #获取方法代码
    # if method_name != "<init>":
    #     code, mapping = mask_java_code(code)
    # 生成注释
    comment = ask_llm(
        model = model,  
        system_prompt_path = "/home/yfx/codecomment/codecomment/methodline/java/new/directly/prompt2.txt",
        temperature = 0.5,    
        max_tokens = 1024,
        code = code,
    )

    return comment

def clean_java_comment(comment):
    """
    清理Java风格的注释，去除星号(*)和多余空格，同时保留注释结构和内容
    
    参数:
        comment (str): 包含Java风格注释的字符串
        
    返回:
        str: 清理后的注释文本
    """
    # 处理空注释
    if not comment or comment.isspace():
        return ""
    
    # 移除注释起始/结束标记
    cleaned = re.sub(r'^\s*/\*\*|\s*\*/\s*$', '', comment, flags=re.MULTILINE)
    # 如果注释以//或///开头，则去除后直接返回
    if re.match(r'^[ \t]*//', cleaned):
        return re.sub(r'^[ \t]*//\/?(?:[ \t])?', '', cleaned)
    
    # 移除每行开头的星号和空格
    lines = []
    for line in cleaned.splitlines():
        # 移除行首星号和后续空格
        line = re.sub(r'^\s*\*+\s*', '', line)
        # 保留空行但移除多余空格
        lines.append(line.strip() if line.strip() else "")
    
    # 重新组合并移除首尾空行
    result = "\n".join(lines).strip()

    # 保留HTML标签和Javadoc标签
    return result

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
def load_prompt(prompt_fp):
    with open(prompt_fp, 'r', encoding='utf-8') as f:
        return f.read()
def ask_llm(
    model: str = "codellama:13B",  
    system_prompt_path = "prompt5.txt",
    temperature: float = 0.5,    
    max_tokens: int = 1024,
    code = "",
) -> str:
    
    client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')
    # client = OpenAI(base_url='https://apivvip.top/v1', api_key='sk-ysFlwsFERsa3SBsRIjHDoNM5nm8feBLwitJCuYdKC5DbAtly')
    # model = "gpt-4o-mini"  
    system_prompt = load_prompt(system_prompt_path)
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content":"Now, generate a high-quality comment for the following Java method:" + code})
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature = temperature,
            max_tokens = max_tokens
        )
    except Exception as e:
        print(f"Error: {e}")

    return response.choices[0].message.content

def save_nodes_to_csv(nodes, file_path):
    """
    将 nodes 字典中的内容保存到 CSV 文件中。
    :param nodes: 包含 Node 对象的字典
    :param file_path: CSV 文件路径
    """
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入每行数据
        for node_name, node in nodes.items():
            # 确保 parameters 是字符串列表
            # print(node)
            if node==None:
                continue
           
            writer.writerow([
                node.path,
                node.name,
                node.full_name,
                node.start_line,
                node.end_line,
                node.comment,
                node.pre_comment,
                node.child_name,
                node.domain,
                node.is_inner_method,
                node.node_level
                ])

def main(config):

    dot_file = config["dot_file"] # 存储调用图地址
    src_path = config["src_path"] # 项目地址
    csv_file_path1 = config["directly_path"] # 存储结果数据地址
    comment_file_path = config["comment_file_path"] # 存储原方法注释地址
    model = config.get("model")  # 使用的模型
    # 方法调用的顺序
    cycle_info , lenght = reverse_topological_sort(dot_file)
    pre_comments = get_precomment(comment_file_path)
    nodes_info = parse_dot_graph(dot_file)
    i=0
    nodes = {}
    level_counts = count_nodes_per_level(cycle_info) # 统计每个层级节点数量
    parts = divide_levels(level_counts) # 将层级划分为4部分

    for result in tqdm(cycle_info):
        i+=1 
        node = result[0] 
        node_level = result[1]          
        path = nodes_info[node]["path"]
        nodes[node] = None
        prefix, method_sig = node.split(':')
        method_name = method_sig.split('(')[0]
        if i<=lenght:
            pass
        else:
            if path:
                file_path = src_path + path + ".java"
                os.path.join(file_path)
                if os.path.exists(file_path):
                    pre_comment = pre_comments.get(node, "")
                    prefix, method_sig = node.split(':')
                    method_name = method_sig.split('(')[0]
                    comment = process(file_path, node, nodes_info, method_name,model)
                    # print(comment)
                    nodes[node] = Node(file_path, method_name, node, nodes_info[node]["start_line"], nodes_info[node]["end_line"], None, None, comment, pre_comment, True, node_level)
                    
 
    if not os.path.exists(csv_file_path1):
        with open(csv_file_path1, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(["file_path", "Name","full_name", "Start Line", "End Line", "Comment", "Pre_Comment", "child Name", "domain", "inner_method", "node_level"])
    else:
        # 检查文件是否为空
        if os.path.getsize(csv_file_path1) == 0:
            with open(csv_file_path1, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # 写入表头
                writer.writerow(["file_path",  "Name","full_name", "Start Line", "End Line", "Comment", "Pre_Comment", "child Name", "domain", "inner_method", "node_level"])
        else:
            # print("文件不为空，跳过写入表头")
            pass
    save_nodes_to_csv(nodes, csv_file_path1)

# 示例调用
if __name__ == "__main__":

    config_str = sys.argv[1]
    config = json.loads(config_str)
    main(config)