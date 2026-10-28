"""合规闸门：规则引擎，不走 LLM。

设计判断：
- 合规检测必须是**确定性**的。LLM 会漏、会解释、会自己改口，而广告法违禁词是
  一张确定的词表，漏一个就是一次投放事故。
- 规则引擎还能给出**精确位置**（第几段第几个字），策划才能一处一处改；
  LLM 只能说"这句有问题"，落不了地。
- LLM 只在「怎么改」这一步介入——那是需要语感的地方。

分工：规则负责"发现"，模型负责"改写"，人负责"过审"。
"""
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "brand_assets.json"

# 把"第一次""第一步"这类正常用法从绝对化用语里排除，避免误报。
_EXCLUDE = {
    ("AD-01", "第一"): r"第一(次|天|周|个|步|眼|时间|阶段|感觉|瓶|次洗|晚|层|件|支|回|遍|款|句|招|条|点|段)",
    ("AD-01", "绝对"): r"绝对值",
    ("AD-01", "无添加"): r"无添加(糖|油)",
    ("AD-02", "修复"): r"修复(包|袋|工|件)",
}


def load_rules():
    return json.loads(DATA.read_text(encoding="utf-8"))


def audit(text: str):
    rules = load_rules()["compliance_rules"]
    hits = []
    for rule in rules:
        for pat in rule["patterns"]:
            exclude = _EXCLUDE.get((rule["id"], pat))
            for m in re.finditer(re.escape(pat), text):
                if exclude:
                    window = text[m.start() : m.start() + len(pat) + 3]
                    if re.search(exclude, window):
                        continue
                hits.append(
                    {
                        "rule_id": rule["id"],
                        "rule_name": rule["name"],
                        "level": rule["level"],
                        "basis": rule["basis"],
                        "span": pat,
                        "position": m.start(),
                        "context": text[max(0, m.start() - 12) : m.end() + 12],
                        "suggestion": rule["suggestion"],
                    }
                )
    hits.sort(key=lambda h: (h["level"] != "违规", h["position"]))

    violations = [h for h in hits if h["level"] == "违规"]
    risks = [h for h in hits if h["level"] == "风险"]
    return {
        "hits": hits,
        "violation_count": len(violations),
        "risk_count": len(risks),
        "passed": len(violations) == 0,
        "total": len(hits),
    }


def style_check(text: str, voice: list[str]):
    """品牌调性的可测部分：感叹号密度、套话句式、段落长度。"""
    issues = []
    bang = text.count("!")
    if bang >= 4:
        issues.append(f"感叹号 {bang} 个，品牌调性要求不堆感叹号")
    for cliche in ["姐妹们", "谁懂啊", "yyds", "闭眼入", "无限回购", "真的绝了"]:
        if cliche in text:
            issues.append(f"出现套话「{cliche}」，多篇重复会触发同质化判定")
    paras = [p for p in text.split("\n") if p.strip()]
    long_paras = [p for p in paras if len(p) > 160]
    if long_paras:
        issues.append(f"有 {len(long_paras)} 段超过 160 字，小红书正文需要更短的呼吸感")
    return issues


def coverage(text: str, assets: list[dict]):
    """卖点覆盖：命中了哪些资产。用词面命中的保守估计，宁可少算。"""
    used, missed = [], []
    for a in assets:
        keys = [a["title"].split("：")[0]] + a.get("tags", [])[:4]
        hit = any(k and k in text for k in keys)
        (used if hit else missed).append({"id": a["id"], "title": a["title"]})
    return used, missed


if __name__ == "__main__":
    demo = "这是最好用的沐浴油，能修复屏障，7天见效，加我微信领券！姐妹们冲！"
    r = audit(demo)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print(style_check(demo, []))
