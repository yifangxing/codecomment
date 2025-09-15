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

def evaluate_bleu_rouge(refs, gens):

    assert len(refs) == len(gens), "两个文件行数不一致！"

    smoothie = SmoothingFunction().method4
    rouge = Rouge()

    bleu_scores = []
    rouge_l_scores = []

    for r, g in zip(refs, gens):
        # 计算 BLEU-4
        bleu = sentence_bleu([r.split()], g.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        bleu_scores.append(bleu)

        # 计算 ROUGE-L
        rouge_score = rouge.get_scores(g, r)[0]["rouge-l"]["f"]
        rouge_l_scores.append(rouge_score)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    print(f"平均 BLEU-4 分数: {avg_bleu:.4f}")
    print(f"平均 ROUGE-L 分数: {avg_rouge_l:.4f}")

    return round(avg_bleu, 4), round(avg_rouge_l, 4)

def compute_bertscore(gen_sentences, ref_sentences):
    # 计算BERTScore得分
    (P, R, F), hashname = score(gen_sentences, ref_sentences, lang="en", return_hash=True)
    return (round(P.mean().item(), 4),round(R.mean().item(), 4),round(F.mean().item(), 4)),hashname

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
    class_doc_file_path ="test.json"
    with open(class_doc_file_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)

    # 取平均值
    avg_score = sum(scores) / len(scores)

    return  round(avg_score, 4)

def main(config):
    hyp_address = config["hyp_address"]
    ref_address = config["ref_address"]
    csv_path = config["csv_path"]
    # 读取 txt 文件并去掉行号
    gen_sentences = read_sentences(hyp_address)
    ref_sentences = read_sentences(ref_address)
    avg_bleu, avg_rouge_l = evaluate_bleu_rouge(ref_sentences, gen_sentences)
    (P, R, F), hashname = compute_bertscore(gen_sentences, ref_sentences)
    # sentencebert1 = compute_sentencebert("sentence-transformers/all-MiniLM-L6-v2", gen_sentences, ref_sentences)
    sentencebert2 = compute_sentencebert("sentence-transformers/all-mpnet-base-v2", gen_sentences, ref_sentences)
    # 写入 CSV（追加）
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["hyp_address", "hash", "bleu-4", "rouge_l", "P", "R", "F", "sentencebert2"])
        writer.writerow([hyp_address, hashname, avg_bleu, avg_rouge_l, P, R, F, sentencebert2])
    print(f"评估结果保存到文件 {csv_path}")


# 示例调用
if __name__ == "__main__":
    config_str = sys.argv[1]
    config = json.loads(config_str)
    main(config)
