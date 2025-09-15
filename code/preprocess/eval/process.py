import os
import csv
import re
import pandas as pd
from tqdm import tqdm
from util import multiline_to_single_line, write_to_dot


def clean_java_comment(comment):
    """
    清理Java风格的注释，去除星号(*)和多余空格，取第一句为注释。
    
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
        cleaned = re.sub(r'^[ \t]*//\/?(?:[ \t])?', '', cleaned)
    
    # 移除每行开头的星号和空格
    lines = [re.sub(r'^\s*\* ?', '', line).strip() for line in cleaned.strip().split('\n')]
    
    # 重新组合并移除首尾空行
    result = "\n".join(lines).strip()

      # 找到第一个句号
    dot_index = result.find('.')
    if dot_index != -1:
        return result[:dot_index + 1].strip()
    else:
        # 没有句号就取第一行
        return lines[0] if lines else ''
    
def multiline_to_single_line(text: str) -> str:
    """
    将多行字符串转换为单行。
    """
    return re.sub(r'\s+', ' ', text).strip()


def extract_java_comment(comment_text: str) -> str:
    """
    从可能被 ```java ``` 或者``` ```包裹的注释中提取实际内容。
    
    参数:
        comment_text: 原始注释文本，可能包含包裹标记
        
    返回:
        提取后的纯净注释内容
    """
    pattern = r'```java\s*(.*?)\s*```'
    
    match = re.search(pattern, comment_text, re.DOTALL)
    
    if match:
        # 提取匹配组中的实际内容
        return match.group(1).strip()
    
    pattern = r'```\s*(.*?)\s*```'
    
    match = re.search(pattern, comment_text, re.DOTALL)
    
    if match:
        # 提取匹配组中的实际内容
        return match.group(1).strip()
    
    # 如果没有匹配到包裹标记，直接返回原文本（去除首尾空白）
    return comment_text.strip()


def save_nodes_to_csv(df,list):
    """
    将 nodes 字典中的内容保存到 CSV 文件中。
    :param nodes: 包含 Node 对象的字典
    :param file_path: CSV 文件路径
    """
    start_line = 0  # 假设从第0行开始读取
    print(len(df))
    for idx, row in tqdm(df.iloc[start_line:].iterrows(), total=len(df)-start_line):
        comment = str(row['Comment'])
        pre_comment = str(row['Pre_Comment'])
        bool = row['inner_method']
        if bool == True:
            if pre_comment !="nan" and pre_comment != "":
                pre_comment = clean_java_comment(pre_comment)
                pre_comment = multiline_to_single_line(pre_comment)
                if pre_comment != "" and comment !="nan": 
                    comment = extract_java_comment(comment)
                    comment = clean_java_comment(comment)
                    comment = multiline_to_single_line(comment)
                    write_to_dot(list+"/hyp.comment", comment) #大模型生成的注释
                    write_to_dot(list+"/ref.comment", pre_comment) #参考注释
                    write_to_dot(list+"/id", str(idx+2)) #注释id与源csv文件id的映射，用来检测用的


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='从保存的CSV文件中提取注释并保存到测试文件中')
    parser.add_argument('path', help='要处理的文件或目录路径')
    parser.add_argument('--list', help='保存测试文件目录')
    args = parser.parse_args()
    csv_path=args.path
    df = pd.read_csv(csv_path)
    save_nodes_to_csv(df,args.list)
    
              