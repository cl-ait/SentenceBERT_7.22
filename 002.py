from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import cosine_similarity

# トークナイザーとモデルをロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 類似度を計算する文
sentence1 = "私は大学生です"
sentence2 = "私は大学に通っている学生ではありません"

# トークン化とテンソル形式への変換
inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True, max_length=128)

# モデルを使って文を埋め込みベクトルに変換
with torch.no_grad():
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

# 埋め込みベクトルの最初のトークン（[CLS]トークン）のベクトルを取得
embedding1 = outputs1.last_hidden_state[:, 0, :]
embedding2 = outputs2.last_hidden_state[:, 0, :]

# コサイン類似度を計算
similarity = cosine_similarity(embedding1, embedding2)

print(f"類似度: {similarity.item()}")  
