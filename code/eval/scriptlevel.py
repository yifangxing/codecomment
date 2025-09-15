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
    config_file = "/home/yfx/codecomment/codecomment/eval/neweval/configlevel.json"
    with open(config_file, "r", encoding="utf-8") as f:
        configs = json.load(f)
    config = configs[2]   
    len = int(config["length"])
    for _ in range(len):   # _ 表示“用不到”的变量
        print("第", _ + 1, "次")
        with open(config_file, "r", encoding="utf-8") as f:
            configs = json.load(f)
        config = configs[2]   
        config["ref_address"] = config["ref_address"]+str(_ + 1)+".comment"
        config["hyp_address"] = config["hyp_address"]+str(_ + 1)+".comment"
        config["directly_ref_address"] = config["directly_ref_address"]+str(_ + 1)+".comment"
        config["directly_hyp_address"] = config["directly_hyp_address"]+str(_ + 1)+".comment"
        subprocess.run([sys.executable, "levelscore.py", json.dumps(config)], check=True)
