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
    config_file = "/home/yfx/codecomment/codecomment/methodline/java/new/ablation/Excluding_important_score/config.json"
    with open(config_file, "r", encoding="utf-8") as f:
        configs = json.load(f)

    config = configs[0]
    subprocess.run([sys.executable, "javatest.py", json.dumps(config)], check=True)
 
