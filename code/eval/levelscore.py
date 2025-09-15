import csv
import json
import os
import sys
from bert_score import score
from sentence_transformers import SentenceTransformer, util
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

def read_sentences(file_path):
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 去掉前面的数字序号（匹配 "数字+空格"）
            clean_line = re.sub(r"^\d+\s+", "", line.strip())
            if clean_line:  # 避免空行
                sentences.append(clean_line)
    return sentences

def compute_sentencebert(model, gen_sentences, ref_sentences):
    # 计算SentenceBERT得分
    # 选择一个适合代码的模型
    model = SentenceTransformer(model)

    # 编码为向量
    gen_embeddings = model.encode(gen_sentences, convert_to_tensor=True)
    ref_embeddings = model.encode(ref_sentences, convert_to_tensor=True)
    # 计算两两相似度
    scores = []
    for ref_emb, gen_emb in zip(ref_embeddings, gen_embeddings):
        score = util.cos_sim(ref_emb, gen_emb).item()  # 单个分数
        scores.append(score)

    # 这是将这个相似性分数保存到文件的代码，为后续可能需要画图做准备
    # class_doc_file_path ="test2.json"
    # with open(class_doc_file_path, "w", encoding="utf-8") as f:
    #         json.dump(scores, f, ensure_ascii=False, indent=4)

    # 取平均值
    avg_score = sum(scores) / len(scores)

    return  round(avg_score, 4)

def main(config):
    hyp_address = config["hyp_address"]
    ref_address = config["ref_address"]
    directly_hyp_address = config["directly_hyp_address"]
    directly_ref_address = config["directly_ref_address"]
    csv_path = config["csv_path"]
    # 读取 txt 文件并去掉行号
    gen_sentences = read_sentences(hyp_address)
    ref_sentences = read_sentences(ref_address)
    directly_gen_sentences = read_sentences(directly_hyp_address)
    directly_ref_sentences = read_sentences(directly_ref_address)
    sentencebert2 = compute_sentencebert("sentence-transformers/all-mpnet-base-v2", gen_sentences, ref_sentences)
    sentencebert4 = compute_sentencebert("sentence-transformers/all-mpnet-base-v2", directly_gen_sentences, directly_ref_sentences)
    # 写入 CSV（追加）
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["hyp_address","qwen2.5-coder:14b+HAG", "qwen2.5-coder:14b" ])
        writer.writerow([hyp_address, sentencebert2, sentencebert4])
    print(f"评估结果保存到文件 {csv_path}")


# 示例调用
if __name__ == "__main__":
    config_str = sys.argv[1]
    config = json.loads(config_str)
    main(config)
