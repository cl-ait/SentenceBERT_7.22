from sentence_transformers import SentenceTransformer, util

# モデルの読み込み
model = SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens-v2')

# 類似度を計算する文
sentence1 = "私は大学生です"
sentence2 = "私は学生です"

# 文章をベクトルに変換する
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)

# コサイン類似度を計算する
similarity = util.pytorch_cos_sim(embedding1, embedding2)

print(f"類似度: {similarity.item()}")