import os
import csv
import re
import pandas as pd
from tqdm import tqdm
from util import multiline_to_single_line, write_to_dot


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
        level = row['node_level']
        if bool == True:
            if pre_comment !="nan" and pre_comment != "":
                pre_comment = clean_java_comment(pre_comment)
                pre_comment = multiline_to_single_line(pre_comment)
                if pre_comment != "" and comment !="nan": 
                    comment = extract_java_comment(comment)
                    comment = clean_java_comment(comment)
                    comment = multiline_to_single_line(comment)
                    write_to_dot(list+"/hyp"+str(level), comment) #大模型生成的注释
                    write_to_dot(list+"/ref"+str(level), pre_comment) #参考注释
                    write_to_dot(list+"/id"+str(level), str(idx+2)) #注释id与源csv文件id的映射，用来检测用的


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='从保存的CSV文件中提取注释并保存到测试文件中')
    parser.add_argument('path', help='要处理的文件或目录路径')
    parser.add_argument('--list', help='保存测试文件目录')
    args = parser.parse_args()
    csv_path=args.path
    df = pd.read_csv(csv_path)
    save_nodes_to_csv(df,args.list)