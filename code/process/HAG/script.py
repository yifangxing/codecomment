# import subprocess

# def run_script(path, config_file):
#     print(f"开始运行 {path} …")
#     completed = subprocess.run(["python", path, "--config", config_file], text=True)
#     if completed.returncode != 0:
#         raise RuntimeError(f"{path} 运行失败，返回码 {completed.returncode}")
#     print(f"{path} 运行完毕")

# if __name__ == "__main__":
#     config_file = "/home/yfx/codecomment/codecomment/methodline/java/new/二次不提供数据流/config.json"
#     # run_script("/home/yfx/codecomment/codecomment/methodline/java/new/二次不提供数据流/javatest.py", config_file)
#     # run_script("/home/yfx/codecomment/codecomment/methodline/java/new/directly/javatest.py", config_file)
#     # run_script("/home/yfx/codecomment/codecomment/methodline/java/new/二次不提供数据流/javatest1.py", config_file)
#     # run_script("/home/yfx/codecomment/codecomment/methodline/java/new/directly/test.py", config_file)
#     run_script("/home/yfx/codecomment/codecomment/methodline/java/new/二次不提供数据流/test.py", config_file)
    

import json
import subprocess
import sys
import shlex   # 可选：自动转义空格/特殊字符
def run_script(path, config_file):
    print(f"开始运行 {path} …")
    completed = subprocess.run(["python", path, "--config", config_file], text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{path} 运行失败，返回码 {completed.returncode}")
    print(f"{path} 运行完毕")

if __name__ == "__main__":
    config_file = "/home/yfx/codecomment/codecomment/methodline/java/new/二次不提供数据流/config1.json"
    with open(config_file, "r", encoding="utf-8") as f:
        configs = json.load(f)

    # # 1) 把命令拆成列表（推荐，避免 shell 注入）
    # cmd = [
    #     '/usr/lib/jvm/java-17-openjdk-amd64/bin/java',
    #     '-Xmx20g',
    #     '-cp',
    #     '/home/yfx/codecomment/codecomment/preprocess/java/callgraph/graph-processor-1.0-SNAPSHOT-jar-with-dependencies.jar',
    #     'org.example.MainRunner',
    #     '/home/yfx/codecomment/codecomment/preprocess/java/callgraph/config.json'
    # ]

    # # 2) 运行并等待结束
    # try:
    #     subprocess.run(cmd, check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f'Java 进程返回非 0：{e.returncode}')

    # 这里随便选一个，比如取第 0 个
    config = configs[0]
    # 把 config 转为字符串传给子脚本
    # 用 json.dumps 保证格式正确
    # 先获取类的功能描述
    subprocess.run([sys.executable, "class_doc.py"], check=True)
    subprocess.run([sys.executable, "test1.py", json.dumps(config)], check=True)
    config = configs[1]
    subprocess.run([sys.executable, "/home/yfx/codecomment/codecomment/methodline/java/new/directly/test.py", json.dumps(config)], check=True)
    config = configs[2]
    subprocess.run([sys.executable, "test1.py", json.dumps(config)], check=True)
    config = configs[3]
    subprocess.run([sys.executable, "/home/yfx/codecomment/codecomment/methodline/java/new/directly/test.py", json.dumps(config)], check=True)
    config = configs[4]
    subprocess.run([sys.executable, "test1.py", json.dumps(config)], check=True)
    config = configs[5]
    subprocess.run([sys.executable, "/home/yfx/codecomment/codecomment/methodline/java/new/directly/test.py", json.dumps(config)], check=True)
