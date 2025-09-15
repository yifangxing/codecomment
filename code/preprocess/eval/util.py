import re
import pandas as pd
from tqdm import tqdm

def extract_docstring(code_str):
    """提取给定代码片段的注释"""
    # 正则表达式匹配三引号包围的多行字符串
    # print("Searching for docstring in:\n", code_str)
    match = re.search(r'"""(.*?)"""', code_str, re.DOTALL)
    if match:
        # 返回匹配的注释内容，并去除首尾的空白字符
        match= match.group(1).strip()
        # match=extract_first_sentence(match)
        return match
    return ""  # 如果没有找到注释，返回

def extract_first_sentence(docstring):
    """提取多行注释的第一句话为方法注释"""
    # 按行分割注释字符串
    lines = docstring.split('\n')
    
    # 找到第一个不为空的行
    start_index = next((i for i, line in enumerate(lines) if line.strip()), None)
    
    if start_index is None:
        return ""  # 如果没有找到不为空的行，返回空字符串
    
    # 从第一个不为空的行开始，继续往下找，直到找到空行（如果有），然后提取空行往上的内容
    for end_index in range(start_index, len(lines)):
        if lines[end_index].strip() == '':
            break
    else:
        end_index = len(lines)  # 如果没有找到空行，取到最后一行
    
    # 提取第一个不为空的行到空行（如果有）之间的内容
    extracted_sentence = '\n'.join(lines[start_index:end_index]).strip()
    
    return extracted_sentence   

def multiline_to_single_line(text: str) -> str:
    """
    将多行字符串转换为单行.
    """
    return re.sub(r'\s+', ' ', text).strip()

def extract_before_example(comment):
    """去除注释中举例子的部分"""
    # 定义匹配模式（不区分大小写）
    pattern = re.compile(
        r'(.*?)\s*(?:e\.g|E\.g|For example|for example|Eg|eg)[,:.]?\s.*',
        re.IGNORECASE | re.DOTALL
    )
    
    match = pattern.fullmatch(comment)
    if match:
        return match.group(1).strip()
    return comment.strip()

def write_to_dot(file_path, content):
    """
    将内容写入 DOT 文件，并在行首添加行号。
    
    :param file_path: DOT 文件路径
    :param content: 要写入的内容
    :return: 返回写入的行号
    """
    try:
        # 首先读取当前文件的行数来确定新行号
        with open(file_path, 'r', encoding='utf-8') as file:
            line_number = sum(1 for _ in file) + 1  # 计算当前行数+1
        
        # 以追加模式写入带行号的内容
        with open(file_path, 'a', encoding='utf-8') as file:
            line_content = f"{line_number} {content}\n"  # 格式: "行号 内容"
            file.write(line_content)
        
        return line_number  # 返回写入的行号
    
    except FileNotFoundError:
        # 如果文件不存在，从行号1开始
        with open(file_path, 'w', encoding='utf-8') as file:
            line_content = f"1 {content}\n"  # 第一行
            file.write(line_content)
        return 1