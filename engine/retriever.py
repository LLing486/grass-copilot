"""品牌资产检索：中文 char-bigram BM25。

为什么不用向量库：
1. 品牌资产只有几十到几百条，稀疏检索的召回在这个规模上不输向量检索；
2. 命中项和命中词可以直接给人看，策划能判断「它为什么召回了这条」——这是
   一个内容团队愿意信任系统的前提；
3. 零额外依赖、零额外延迟、零 embedding 成本。
生产环境可升级为 hybrid（BM25 + embedding），但 BM25 应保留作为召回基线。
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "brand_assets.json"

_WORD = re.compile(r"[a-zA-Z]+|\d+")
_CN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str):
    """中文取 1-gram + 2-gram，英文数字取词。"""
    text = text or ""
    tokens = [w.lower() for w in _WORD.findall(text)]
    for seg in _CN.findall(text):
        tokens.extend(seg)  # unigram
        tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))  # bigram
    return tokens


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.tf = [Counter(t) for t in self.doc_tokens]
        df = Counter()
        for tf in self.tf:
            for term in tf:
                df[term] += 1
        n = len(docs)
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query, top_k=6):
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for term in set(q):
                if term not in tf:
                    continue
                f = tf[term]
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / self.avgdl
                )
                s += self.idf.get(term, 0) * f * (self.k1 + 1) / denom
            scores.append((s, i))
        scores.sort(reverse=True)
        return [(s, i) for s, i in scores[:top_k] if s > 0]


def load_assets():
    return json.loads(DATA.read_text(encoding="utf-8"))


def build_index(assets):
    # 标题和标签加权：同一段文本重复一次，等同于提权。
    docs = [
        f"{a['title']} {a['title']} {a.get('text','')} {' '.join(a.get('tags',[])) * 2}"
        for a in assets
    ]
    return BM25(docs)


def retrieve(query, top_k=6, min_score=1.0):
    data = load_assets()
    assets = data["assets"]
    bm = build_index(assets)
    hits = []
    for score, idx in bm.search(query, top_k=top_k * 2):
        if score < min_score:
            continue
        a = assets[idx]
        hits.append(
            {
                "id": a["id"],
                "type": a["type"],
                "title": a["title"],
                "text": a["text"],
                "evidence": a.get("evidence"),
                "tags": a.get("tags", []),
                "score": round(score, 2),
                "why": _match_terms(query, assets[idx]),
            }
        )
        if len(hits) >= top_k:
            break
    return hits


def _match_terms(query, asset):
    q = set(tokenize(query))
    pool = asset["title"] + " " + asset.get("text", "") + " " + " ".join(
        asset.get("tags", [])
    )
    shared = sorted({t for t in set(tokenize(pool)) if t in q and len(t) >= 2})
    return shared[:6]


if __name__ == "__main__":
    for h in retrieve("干皮 秋冬 洗完紧绷 起皮 换什么沐浴产品"):
        print(h["score"], h["id"], h["title"], h["why"])
