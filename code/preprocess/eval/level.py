from collections import Counter
import os
import csv
import re
import pandas as pd
from tqdm import tqdm
from util import multiline_to_single_line, write_to_dot
import networkx as nx

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

def multiline_to_single_line(text: str) -> str:
    """
    将多行字符串转换为单行.
    """
    return re.sub(r'\s+', ' ', text).strip()


def extract_java_comment(comment_text: str) -> str:
    """
    从大模型输出中提取 Java 方法的注释部分（支持 ```java``` 包裹 + /** */ JavaDoc 格式）。
    
    参数:
        comment_text: 原始注释文本，可能包含包裹标记或JavaDoc
        
    返回:
        提取后的纯净 JavaDoc 注释内容
    """
    # 优先匹配 ```java ... ```
    pattern = r'```java\s*(.*?)\s*```'
    match = re.search(pattern, comment_text, re.DOTALL)
    if match:
        code_block = match.group(1).strip()
    else:
        # 再尝试匹配 ``` ... ```
        pattern = r'```\s*(.*?)\s*```'
        match = re.search(pattern, comment_text, re.DOTALL)
        if match:
            code_block = match.group(1).strip()
        else:
            code_block = comment_text.strip()

    # 在提取出的代码块中寻找 JavaDoc 注释 /** ... */
    javadoc_pattern = r'(/\*\*.*?\*/)'
    javadoc_match = re.search(javadoc_pattern, code_block, re.DOTALL)
    if javadoc_match:
        return javadoc_match.group(1).strip()

    # 如果没有 JavaDoc，则直接返回原始代码块
    return code_block

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

    # 计算每个节点的出度和入度
    out_degree = dict(G.out_degree())
    in_degree = dict(G.in_degree())

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

    # 存储每个节点的孩子节点
    children_map = {node: list(G.successors(node)) for node in G.nodes()}

    return result

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
    remaining_levels = remaining_levels[::-1]
    # 剩余层级总数
    total_nodes = sum(level_counts[lvl] for lvl in remaining_levels)
    target = total_nodes / 3  # 每部分目标节点数

    parts = []
    current_sum = 0
    start = remaining_levels[0]
    count_parts = 0

    total_lv = len(remaining_levels)
    for i, lvl in enumerate(remaining_levels):
        current_sum += level_counts[lvl]
        # 计算“后面还剩几层”
        left = total_lv - i - 1

        # 第一段：切完后至少留 2 层
        if count_parts == 0 and current_sum >= target and left >= 2:
            end = lvl
            parts.append((end, start))
            count_parts += 1
            start = remaining_levels[i+1]
            current_sum = 0
            continue

        # 第二段：切完后至少留 1 层
        if count_parts == 1 and current_sum >= target and left >= 1:
            end = lvl
            parts.append((end, start))
            count_parts += 1
            start = remaining_levels[i+1]
            current_sum = 0
            continue

        # 第二段：如果只剩两层，确保留下最后一层
        if count_parts == 1 and left == 1:
            end = lvl
            parts.append((end, start))
            count_parts += 1
            start = remaining_levels[i+1]
            current_sum = 0

        # 最后一层：强制收尾
        if i == total_lv - 1:
            end = lvl
            parts.append((end, start))
    parts = parts[::-1]
    return [part1] + parts

def count_nodes_per_level(result):
    """
    统计每个层级的节点总数。
    """
    level_counts = Counter()

    for _, level in result:
        level_counts[level] += 1

    return dict(level_counts)

def save_nodes_to_csv(df,dot_path,list):
    """
    将 nodes 字典中的内容保存到 CSV 文件中。
    :param nodes: 包含 Node 对象的字典
    :param file_path: CSV 文件路径
    """
    start_line = 0  # 假设从第0行开始读取
    result = reverse_topological_sort(dot_path)
    level_counts = count_nodes_per_level(result)
    parts = divide_levels(level_counts)
    print(len(df))
    for idx, row in tqdm(df.iloc[start_line:].iterrows(), total=len(df)-start_line):
        comment = str(row['Comment'])
        pre_comment = str(row['Pre_Comment'])
        bool = row['inner_method']
        node_level = row['node_level']
        if bool == True:
            if pre_comment !="nan" and pre_comment != "":
                pre_comment = clean_java_comment(pre_comment)
                pre_comment = multiline_to_single_line(pre_comment)
                if pre_comment != "" and comment !="nan": 
                    comment = extract_java_comment(comment)
                    comment = clean_java_comment(comment)
                    comment = multiline_to_single_line(comment)
                    if node_level == 1 :
                        continue
                    if node_level == 2 :
                        level = "1"
                    elif node_level >= parts[1][0] and node_level <= parts[1][1]:
                        level = "2"
                    elif node_level >= parts[2][0] and node_level <= parts[2][1]:
                        level = "3"
                    elif node_level >= parts[3][0] and node_level <= parts[3][1]:
                        level = "4"
                    write_to_dot(list+"/hyp2"+str(level)+".comment", comment) #大模型生成的注释
                    write_to_dot(list+"/ref2"+str(level)+".comment", pre_comment) #参考注释
                    write_to_dot(list+"/id2"+str(level), str(idx+2)) #注释id与源csv文件id的映射，用来检测用的


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='从保存的CSV文件中提取注释并保存到测试文件中')
    parser.add_argument('path', help='要处理的文件或目录路径')
    parser.add_argument('--graph', help='调用图地址')
    parser.add_argument('--out', help='保存结果地址')
    args = parser.parse_args()
    csv_path=args.path
    df = pd.read_csv(csv_path)
    save_nodes_to_csv(df,args.graph,args.out)