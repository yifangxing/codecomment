import csv
import json
import os
import random
import re
from pathlib import Path
from collections import defaultdict
from string import Template
from typing import Dict, Optional
import javalang
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
from openai import OpenAI
import requests

def get_selfloop_edges(G):
    """兼容不同NetworkX版本的自环边获取方法"""
    # NetworkX 2.1+版本支持selfloop_edges
    if hasattr(G, 'selfloop_edges'):
        return list(G.selfloop_edges())
    # 旧版本手动检测自环边
    return [(u, v) for u, v in G.edges() if u == v]

def remove_edges_random(G):
    """随机删除环结构中一条边"""
    NG = G.copy()
    sccs = [c for c in nx.strongly_connected_components(NG) if len(c) > 1]
    NG.remove_edges_from(get_selfloop_edges(NG)) # 删除自环边
    for scc in sccs:
        # 提取强连通分量内的边
        edges_in_scc = [(u, v) for u in scc for v in NG.successors(u) if v in scc]
        if edges_in_scc:
            # 随机选择并删除一条边
            NG.remove_edge(*random.choice(edges_in_scc))
    return NG

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

    return result, length, children_map, in_degree

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
            
def save_node_to_csv(node, file_path):
    """
    将单个 Node 对象追加保存到 CSV 文件中。
    :param node: Node 对象
    :param file_path: CSV 文件路径
    """
    if node is None:
        return  # 跳过空节点
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
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

def find_child_comment(nodes, child_nodes, order):
    """
    获取每个节点的孩子结点的注释。
    """
    template = []
    method_names = {}
    if(child_nodes==None):
            return []
    else:
        # 存储每个方法的名称和其在整个项目中的完整命名的映射
        for child in child_nodes:
            prefix, method_sig = child.split(':')
            method_name = method_sig.split('(')[0]
            method_names[method_name] = child
        # 根据源代码中的调用的顺序，保证注释的顺序与代码调用顺序一致
        for method_name in order:
            # print("method_name",method_name)
            full_name = method_names[method_name]
            node_info = nodes[full_name]
            prefix, method_sig = child.split(':')
            method_name = method_sig.split(')')[0]+")"
            if node_info==None:
                # child_node不是方法
                continue
            else:
                comment = clean_java_comment(node_info.comment)
                template.append((method_name+":"+comment))
    # print(template) 
    return template

def find_child_comments(nodes, child_nodes, order, mapping):
    """
    获取每个节点的孩子结点的注释。
    """
    template = []
    method_names = {}
    if(child_nodes==None):
            return []
    else:
        # 存储每个方法的名称和其在整个项目中的完整命名的映射
        for child in child_nodes:
            prefix, method_sig = child.split(':')
            method_name = method_sig.split('(')[0]
            method_names[method_name] = child
        # 根据源代码中的调用的顺序，保证注释的顺序与代码调用顺序一致
        for method_name in order:
            # print("method_name",method_name)
            full_name = method_names[method_name]
            node_info = nodes[full_name]
            prefix, method_sig = child.split(':')
            method_name = method_sig.split(')')[0]+")"
            if node_info==None:
                # child_node不是方法
                continue
            else:
                comment = clean_java_comment(node_info.comment)
                if method_name in mapping:
                    method_name = mapping[method_name]
                template.append((method_name+":"+comment))
    # print(template) 
    return template

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
    
def get_comment(node, nodes, method_name, child_nodes, method_code , order, score, model):
    """
    分本节点是否是叶子节点，返回不同情况处理后节点的注释。
    """
    callees_docstrings = []
    processed_response = ""
    if(child_nodes == None and method_code == None):
        # 外部库节点
        processed_response = ask_ollama_ext_node(
            model = model,  
            system_prompt_path = "/home/yfx/codecomment/codecomment/methodline/java/new/ablation/Remove_dynamic_prompt/system_prompt.txt",
            temperature = 0.5,
            max_tokens = 1024, 
            method = method_name
            )
        processed_response = extract_java_comment(processed_response)
        return processed_response
    
    elif(child_nodes == None and method_code != None ):
        # 项目中的叶子节点
        processed_response = ask_ollama_leaf_node_domain(
            model = model,  
            system_prompt_path = "/home/yfx/codecomment/codecomment/methodline/java/new/ablation/Remove_dynamic_prompt/system_prompt.txt",
            user_prompt_path = "/home/yfx/codecomment/codecomment/methodline/java/new/ablation/Remove_dynamic_prompt/user_prompt1.txt",
            temperature = 0.5,
            max_tokens = 1024, 
            code = method_code
            )
        processed_response = extract_java_comment(processed_response)
        return processed_response
    else: 
        # 如果有孩子节点，获取孩子节点的注释
        callees_docstrings = find_child_comment(nodes, child_nodes, order)
        processed_response = ask_ollama_non_domain(
            model = model,  
            system_prompt_path = "/home/yfx/codecomment/codecomment/methodline/java/new/ablation/Remove_dynamic_prompt/system_prompt.txt",
            user_prompt_path = "/home/yfx/codecomment/codecomment/methodline/java/new/ablation/Remove_dynamic_prompt/user_prompt.txt",
            temperature = 0.5,    
            max_tokens = 1024,
            code = method_code,
            callees_docstrings = callees_docstrings,
            score = score
        )
        processed_response = extract_java_comment(processed_response)
        return processed_response
    
def load_prompt(prompt_fp):
    with open(prompt_fp, 'r', encoding='utf-8') as f:
        return f.read()

def ask_ollama_non_domain(
    model: str = "qwen2.5-coder:14b", 
    system_prompt_path = "" ,
    user_prompt_path = "prompt7.txt",
    temperature: float = 0.5,    
    max_tokens: int = 1024,
    code = "",
    callees_docstrings = "" ,
    score = "" 
) -> str:
    # client = OpenAI(base_url='https://apivvip.top/v1', api_key='sk-ysFlwsFERsa3SBsRIjHDoNM5nm8feBLwitJCuYdKC5DbAtly')
    # model = "gpt-4o-mini" 
    client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')
    prompt_template = load_prompt(user_prompt_path)
    user_prompt = prompt_template.replace('{{code}}', json.dumps(code)).replace('{{callees_docstrings}}', json.dumps(callees_docstrings)).replace('{{score}}', json.dumps(score))
    system_prompt = load_prompt(system_prompt_path)
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

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

def ask_ollama_ext_node(
    model: str = "qwen2.5-coder:14b ",  
    system_prompt_path = "prompt2.txt",
    temperature: float = 0.5,    
    max_tokens: int = 1024,
    method = None,
) -> str:
    
    client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')
    # client = OpenAI(base_url='https://apivvip.top/v1', api_key='sk-ysFlwsFERsa3SBsRIjHDoNM5nm8feBLwitJCuYdKC5DbAtly')
    # model = "gpt-4o-mini" 
    system_prompt = load_prompt(system_prompt_path)
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content":"Now, generate a high-quality comment for the following external method: " + method})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature = temperature,
            max_tokens = max_tokens
        )
    except Exception as e:
        print(f"Error: {e}")
        return ""
    return response.choices[0].message.content

def ask_ollama_leaf_node_domain(
    model: str = "qwen2.5-coder:14b", 
    system_prompt_path = "prompt4.txt", 
    user_prompt_path = "prompt6.txt",
    temperature: float = 0.5,    
    max_tokens: int = 1024,
    code = None,
) -> str:
    # client = OpenAI(base_url='https://apivvip.top/v1', api_key='sk-ysFlwsFERsa3SBsRIjHDoNM5nm8feBLwitJCuYdKC5DbAtly')
    # model = "gpt-4o-mini" 
    client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')
    prompt_template = load_prompt(user_prompt_path)
    user_prompt = prompt_template.replace('{{code}}', code)
    system_prompt = load_prompt(system_prompt_path)
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    # print(user_prompt)
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