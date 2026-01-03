import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer
import torch.optim as optim
from sentence_transformers import SentenceTransformer, util
from sacrebleu import corpus_bleu
from tqdm import tqdm

# ===================== 1. 全局配置 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"使用设备: {DEVICE}")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 数据集参数
TRAIN_SIZE = 900  # 原版低资源设定：900句训练
TEST_SIZE = 100   # 100句测试
SENT_LEN_MIN = 5  # 最小句子长度（词数）
SENT_LEN_MAX = 20 # 最大句子长度（词数）

# 训练参数
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NOISE_BASE = 0.05
NOISE_STEP = 0.01
NOISE_MAX = 0.09
ALPHA = 0.5  # 翻译损失:信道损失 = 0.5:0.5

# 模型与评估
MODEL_NAME = "./opus-mt-en-fr"  # 原版常用基线模型
SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2').to(DEVICE)
PROMPT_TEMPLATES = {
    # 原版基线模板
    "original": ["Translate the following English to French: {text}"],
    # 改进1：分层模板（短/中/长句）
    "improved1": {
        "short": ["Translate English short sentence to French: {text}"],
        "medium": ["Accurately translate English sentence to French: {text}"],
        "long": ["Translate English long sentence to French, keep structure: {text}"]
    },
    # 改进3：融合词典的模板
    "improved3": ["Translate {text} to French, use fixed collocation: {colloc}"]
}
# 法语常用搭配词典（改进3用）
FRENCH_COLLOC = {
    "programming": "la programmation",
    "hello": "bonjour",
    "test": "le test",
    "love": "aimer",
    "world": "le monde"
}

# ===================== 2. 数据集下载与预处理 =====================
def download_tatoeba_data():
    """下载并预处理Tatoeba英-法平行语料（对齐原版论文）"""
    if not os.path.exists("sentences.csv") or not os.path.exists("links.csv"):
        raise FileNotFoundError("请先下载sentences.csv和links.csv到当前目录！")
    
    # 加载数据（核心修复：跳过错误行）
    sentences = pd.read_csv(
        "sentences.csv", 
        sep="\t", 
        names=["id", "lang", "text"],
        on_bad_lines='skip',
        low_memory=False
    )
    links = pd.read_csv(
        "links.csv", 
        sep="\t", 
        names=["sent_id1", "sent_id2"],
        on_bad_lines='skip',  # 关键：跳过格式错误的行
        low_memory=False
    )
    
    # 筛选英-法句子
    en_sent = sentences[sentences["lang"] == "eng"].set_index("id")
    fr_sent = sentences[sentences["lang"] == "fra"].set_index("id")
    
    # 匹配平行句对
    en_fr_links = links[
        (links["sent_id1"].isin(en_sent.index)) & 
        (links["sent_id2"].isin(fr_sent.index))
    ]
    
    # 构建平行语料
    parallel_data = []
    for _, row in tqdm(en_fr_links.iterrows(), desc="匹配平行语料"):
        try:
            en_text = en_sent.loc[row["sent_id1"], "text"].strip()
            fr_text = fr_sent.loc[row["sent_id2"], "text"].strip()
            # 过滤长度和空值
            en_len = len(en_text.split())
            fr_len = len(fr_text.split())
            if (SENT_LEN_MIN <= en_len <= SENT_LEN_MAX and 
                SENT_LEN_MIN <= fr_len <= SENT_LEN_MAX and 
                en_text and fr_text):
                parallel_data.append({"en": en_text, "fr": fr_text})
        except:
            continue
    
    # 随机划分训练/测试集
    random.shuffle(parallel_data)
    train_data = parallel_data[:TRAIN_SIZE]
    test_data = parallel_data[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
    
    # 保存数据
    def save_data(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(f"{item['en']}\t{item['fr']}\n")
    
    save_data(train_data, "tatoeba_en_fr_train.txt")
    save_data(test_data, "tatoeba_en_fr_test.txt")
    print(f"数据预处理完成！训练集{len(train_data)}句，测试集{len(test_data)}句")
    return train_data, test_data

# ===================== 3. 数据加载与Prompt处理 =====================
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, mode="original", add_noise=False, epoch=0):
        self.data = data
        self.tokenizer = tokenizer
        self.mode = mode
        self.add_noise = add_noise
        self.noise_prob = min(NOISE_BASE + epoch * NOISE_STEP, NOISE_MAX)
    
    def _add_targeted_noise(self, text):
        """改进2：低频词定向噪声（替代语义感知噪声）"""
        words = text.split()
        # 统计词频（简单模拟低频词）
        word_freq = pd.Series(words).value_counts()
        low_freq_words = [w for w in words if word_freq[w] <= 1]
        
        noisy_words = []
        for word in words:
            if word in low_freq_words and random.random() < self.noise_prob:
                # 仅对低频词加噪声（替换为随机字符）
                noisy_words.append(word[:-1] + random.choice("abcdef") if len(word)>1 else word)
            else:
                noisy_words.append(word)
        return " ".join(noisy_words)
    
    def _get_prompt(self, text):
        """生成Prompt（原版/改进1/改进3）"""
        if self.mode == "original":
            template = random.choice(PROMPT_TEMPLATES["original"])
            return template.format(text=text)
        elif self.mode == "improved1":
            # 改进1：分层Prompt（按句子长度）
            sent_len = len(text.split())
            if sent_len <= 8:
                template = random.choice(PROMPT_TEMPLATES["improved1"]["short"])
            elif sent_len <= 15:
                template = random.choice(PROMPT_TEMPLATES["improved1"]["medium"])
            else:
                template = random.choice(PROMPT_TEMPLATES["improved1"]["long"])
            return template.format(text=text)
        elif self.mode == "improved2":
            # 改进2：使用原版Prompt模板（仅加噪声，不改变Prompt格式）
            template = random.choice(PROMPT_TEMPLATES["original"])
            return template.format(text=text)  # 确保使用正确的变量名text
        elif self.mode == "improved3":
            # 改进3：Prompt+词典融合
            colloc = ""
            for word in text.split():
                if word.lower() in FRENCH_COLLOC:
                    colloc = FRENCH_COLLOC[word.lower()]
                    break
            template = random.choice(PROMPT_TEMPLATES["improved3"])
            return template.format(text=text, colloc=colloc)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        en_text = self.data[idx]["en"]
        fr_text = self.data[idx]["fr"]
        
        # 处理Prompt和噪声
        if self.add_noise and self.mode == "improved2":
            en_text = self._add_targeted_noise(en_text)
        prompt_text = self._get_prompt(en_text)
        
        # 编码
        inputs = self.tokenizer(
            text=prompt_text, return_tensors="pt", truncation=True, max_length=128, padding="max_length"
        )
        targets = self.tokenizer(
            fr_text, return_tensors="pt", truncation=True, max_length=128, padding="max_length"
        )
        
        # 构造标签（忽略padding）
        labels = targets["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
            "en_text": en_text,
            "fr_text": fr_text
        }

# ===================== 4. 模型训练与评估 =====================
def train_model(train_data, test_data, mode="original"):
    """训练模型（基线/改进策略）"""
    # 加载模型和tokenizer
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    print(f"\n开始训练：{mode} 模式")
    for epoch in range(EPOCHS):
        model.train()
        train_dataset = TranslationDataset(train_data, tokenizer, mode, add_noise=True, epoch=epoch)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        total_loss = 0.0
        noise_prob = min(NOISE_BASE + epoch * NOISE_STEP, NOISE_MAX)
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            batch_en = batch["en_text"]  # 批次中的英文句子列表
            
            optimizer.zero_grad()
            
            # 翻译损失
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_trans = outputs.loss
            
            # 信道损失（Prompt一致性）- 修复维度匹配问题
            loss_channel = torch.tensor(0.0).to(DEVICE)
            if mode != "improved2":  # 改进2仅加噪声，不计算信道损失
                # 对整个批次生成新Prompt
                prompt2_list = []
                for en_text in batch_en:
                    if mode == "original":
                        prompt2 = random.choice(PROMPT_TEMPLATES["original"]).format(text=en_text)
                    elif mode == "improved1":
                        sent_len = len(en_text.split())
                        if sent_len <= 8:
                            prompt2 = random.choice(PROMPT_TEMPLATES["improved1"]["short"]).format(text=en_text)
                        elif sent_len <= 15:
                            prompt2 = random.choice(PROMPT_TEMPLATES["improved1"]["medium"]).format(text=en_text)
                        else:
                            prompt2 = random.choice(PROMPT_TEMPLATES["improved1"]["long"]).format(text=en_text)
                    else:  # improved3
                        colloc = ""
                        for word in en_text.split():
                            if word.lower() in FRENCH_COLLOC:
                                colloc = FRENCH_COLLOC[word.lower()]
                                break
                        prompt2 = random.choice(PROMPT_TEMPLATES["improved3"]).format(text=en_text, colloc=colloc)
                    prompt2_list.append(prompt2)
                
                # 批次编码，保证维度匹配
                inputs2 = tokenizer(
                    prompt2_list, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=128, 
                    padding="max_length"
                ).to(DEVICE)
                outputs2 = model(
                    input_ids=inputs2["input_ids"], 
                    attention_mask=inputs2["attention_mask"], 
                    labels=labels
                )
                loss_channel = outputs2.loss
            
            # 总损失
            loss = ALPHA * loss_trans + (1 - ALPHA) * loss_channel
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f} (噪声概率: {noise_prob:.2f})")
    
    # 保存模型
    model_path = f"./{mode}_model"
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"训练完成！模型保存至：{model_path}")
    
    # 评估模型
    evaluate_model(model, tokenizer, test_data, mode)

def evaluate_model(model, tokenizer, test_data, mode):
    """评估模型（BLEU/语义保留度/Prompt适配率）"""
    model.eval()
    test_dataset = TranslationDataset(test_data, tokenizer, mode, add_noise=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    refs = []
    preds = []
    en_texts = []
    
    print("\n开始评估...")
    for batch in tqdm(test_loader, desc="生成预测结果"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        en_text = batch["en_text"][0]
        fr_ref = batch["fr_text"][0]
        
        # 生成预测
        with torch.no_grad():
            generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
            fr_pred = tokenizer.decode(generated[0], skip_special_tokens=True)
        
        en_texts.append(en_text)
        refs.append([fr_ref])
        preds.append(fr_pred)
    
    # 计算BLEU分数
    bleu = corpus_bleu(preds, refs)
    bleu_score = bleu.score
    
    # 计算语义保留度（预测vs标准答案）
    ref_embeddings = SEMANTIC_MODEL.encode([r[0] for r in refs], convert_to_tensor=True).to(DEVICE)
    pred_embeddings = SEMANTIC_MODEL.encode(preds, convert_to_tensor=True).to(DEVICE)
    semantic_similarities = util.cos_sim(pred_embeddings, ref_embeddings).diag().cpu().numpy()
    avg_semantic = np.mean(semantic_similarities)
    
    # 计算Prompt适配率
    prompt_match = 0
    for en_text in en_texts:
        if mode == "original":
            prompt_match += 1  # 原版固定模板
        elif mode == "improved1":
            # 分层模板适配：长度匹配则算适配
            sent_len = len(en_text.split())
            if (sent_len <= 8 and "short" in PROMPT_TEMPLATES["improved1"]) or \
               (8 < sent_len <= 15 and "medium" in PROMPT_TEMPLATES["improved1"]) or \
               (sent_len > 15 and "long" in PROMPT_TEMPLATES["improved1"]):
                prompt_match += 1
        elif mode == "improved2":
            prompt_match += 1  # 改进2仅加噪声
        else:  # improved3
            # 词典匹配则算适配
            match = any(word.lower() in FRENCH_COLLOC for word in en_text.split())
            prompt_match += 1 if match else 0
    prompt_rate = (prompt_match / len(en_texts)) * 100
    
    # 综合得分（BLEU*0.4 + 语义保留度*0.4 + 适配率*0.2）
    total_score = bleu_score * 0.4 + avg_semantic * 40 + prompt_rate * 0.2
    
    # 输出结果
    print(f"\n========== 实验结果 ({mode}) ==========")
    print(f"BLEU分数: {bleu_score:.2f}")
    print(f"平均语义保留度: {avg_semantic:.4f}")
    print(f"Prompt适配率: {prompt_rate:.2f}%")
    print(f"综合得分: {total_score:.2f}")
    
    # 输出示例
    print("\n预测示例：")
    for i in range(min(3, len(en_texts))):
        print(f"原文(英): {en_texts[i]}")
        print(f"标准答案(法): {refs[i][0]}")
        print(f"模型预测(法): {preds[i]}")
        print("---")

# ===================== 5. 主函数：执行完整实验 =====================
if __name__ == "__main__":
    # 步骤1：下载预处理数据
    train_data, test_data = download_tatoeba_data()
    
    # 步骤2：复现原版基线
    # train_model(train_data, test_data, mode="original")
    
    # 步骤3：验证改进策略
    # train_model(train_data, test_data, mode="improved1")  # 分层Prompt
    train_model(train_data, test_data, mode="improved2")  # 低频词噪声
    train_model(train_data, test_data, mode="improved3")  # Prompt+词典融合