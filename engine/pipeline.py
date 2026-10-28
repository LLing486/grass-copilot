"""内容流水线：六个角色，每个只做一件事。

链路：Brief 解析 → 检索改写(RAG) → 策略 → 撰写 → 合规闸门 → 改写复核 → 度量

为什么拆成这么多步，而不是一个大 prompt：
1. 可归因。出问题时能定位到"是召回错了、还是策略跑偏、还是文案越界"，而不是
   面对一段"感觉不对"的文本。
2. 可替换。策略和文案可以换模型，闸门必须是规则，边界清晰才敢上线。
3. 可度量。每一步都有中间产物，才谈得上"规则解决率""人工改写率"这些指标。
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import audit as audit_mod
from . import factcheck
from .llm import chat_json
from .retriever import load_assets, retrieve

BANNED = """绝对化用语（最/第一/顶级/极致/唯一/独家/100%/纯天然/零添加/无添加/彻底/绝对）
医疗与功效（修复/治疗/治愈/消炎/杀菌/抗炎/祛痘/止痒/美白/祛斑/防脱）
时效承诺（7天/三天/一周见效/立刻见效）
导流（微信/vx/加我/私信我/链接/扫码）
套话（姐妹们冲/谁懂啊/yyds/闭眼入/无限回购/真的绝了/神器/绝了）"""

SYSTEM = "你是一名资深小红书种草内容策划，服务过美妆个护品牌。你严格按照品牌资产和平台规范工作，不编造功效，不使用套话。"


def _step(name, fn, *args, **kw):
    t0 = time.time()
    out = fn(*args, **kw)
    return out, round(time.time() - t0, 1)


# ---------------------------------------------------------------- 1. Brief 解析
def parse_brief(brief_text: str):
    data = load_assets()
    brand = data["brand"]
    prompt = f"""把下面这段客户 brief 解析成结构化要素。只输出 JSON。

品牌背景（已知，不必重复）：{brand['positioning']}｜人群：{brand['target']}

brief 原文：
{brief_text}

输出字段：
{{
  "goal": "这次内容要达成什么（一句话）",
  "audience": "目标人群，尽量具体到场景",
  "pain": "要打的痛点",
  "key_points": ["必须出现的卖点，最多3条"],
  "avoid": ["这次明确不要说的"],
  "client_demands": ["客户硬塞进来、可能有合规或品牌口径风险的具体说法（功效宣称、绝对化说法、指定话术），原样摘出；普通的执行要求不要放进来；没有就给空数组"],
  "search_queries": ["用来检索品牌知识库的3个查询语句，写成用户会搜的口语，不要用品牌内部术语"]
}}"""
    return chat_json(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
    )


# ---------------------------------------------------------------- 2. RAG 检索
def rag(brief: dict):
    queries = brief.get("search_queries") or [brief.get("pain", "")]
    anchors = [a for a in (brief.get("key_points") or []) if a]
    seen, hits = set(), []

    # 1) 客户点名的主推卖点先落位。BM25 是相似度检索，它对「这次必须说三油配方」
    #    这种业务约束无感——客户明确要求的点，不能交给相似度碰运气。
    if anchors:
        q_tokens = " ".join(anchors)
        for h in retrieve(q_tokens, top_k=8, min_score=0.0):
            if h["id"] in seen:
                continue
            if any(_anchor_hit(a, h) for a in anchors):
                seen.add(h["id"])
                h["from_query"] = "客户指定卖点（锚点召回）"
                h["anchor"] = True
                hits.append(h)

    # 2) 再用 brief 衍生的口语检索式做相似度召回
    for q in queries[:3]:
        for h in retrieve(q, top_k=6):
            if h["id"] in seen:
                continue
            seen.add(h["id"])
            h["from_query"] = q
            hits.append(h)

    hits.sort(key=lambda h: (0 if h.get("anchor") else 1, -h["score"]))
    return queries, hits[:9]


def _anchor_hit(anchor: str, hit: dict):
    probe = anchor.replace("版", "")
    pool = hit["title"] + "".join(hit.get("tags", []))
    return probe in pool or any(t and t in probe for t in hit.get("tags", []))



# ---------------------------------------------------------------- 3. 策略
def strategize(brief: dict, hits: list[dict]):
    assets_txt = "\n".join(
        f"[{h['id']}|{h['type']}|证据:{h['evidence']}] {h['title']}：{h['text']}"
        for h in hits
    )
    prompt = f"""基于 brief 和召回到的品牌资产，出 3 条差异化的内容策略。只输出 JSON。

brief 解析结果：{json.dumps(brief, ensure_ascii=False)}
品牌资产：
{assets_txt}

要求：
- 3 条必须用不同骨架（痛点共鸣 / 避雷干货 / 场景故事），不能是同一个结构换句话
- 每条写清：打法、必须引用的资产 id、不要碰的雷区
- 如果某条卖点证据等级是 weak，不要作为主主张

输出：
{{"strategies":[{{"id":"S1","structure":"痛点共鸣型","angle":"一句话打法","must_cite":["P-01"],"points":["要说的3个点"],"avoid":["这条不能说的"]}}]}}"""
    return chat_json(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=2000,
    )


# ---------------------------------------------------------------- 4. 撰写
def write_one(strategy: dict, brief: dict, hits: list[dict]):
    data = load_assets()
    voice = "\n".join("- " + v for v in data["brand"]["voice"])
    cite = [h for h in hits if h["id"] in strategy.get("must_cite", [])] or hits[:3]
    assets_txt = "\n".join(
        f"[{h['id']}] {h['title']}：{h['text']}" for h in cite
    )
    prompt = f"""按策略写一篇小红书种草笔记。只输出 JSON。

策略：{json.dumps(strategy, ensure_ascii=False)}
目标人群：{brief.get('audience')}
痛点：{brief.get('pain')}
必须出现的卖点：{brief.get('key_points')}
不要出现：{brief.get('avoid')}

可引用的品牌资产（只能用这些，不许编造数字、成分、检测结论）：
{assets_txt}

品牌调性：
{voice}

合规分工：你只负责写得好、写得真、按客户要求交付。**违禁词由下一道闸门负责拦截和改写，不是你的职责**，所以客户点名要的说法照写，不要自己改口径。
客户点名要求的说法（必须出现在正文里）：{brief.get('client_demands') or '无'}

仍然不要碰的：编造数字/成分/检测结论（这一条没有兜底），以及套话堆砌。

写作要求：
- 标题 ≤ 20 字，要有具体场景或具体部位，不要抽象形容词
- 正文 300-450 字，第一人称，分段短，每段一个意思
- 至少出现 2 处只有真的用过才写得出的细节（水温、用量、天数、部位、穿什么衣服）
- 不点名竞品；不承诺时间效果；不喊人冲
- 结尾不要导流，自然收在体验上

输出：
{{"title":"","body":"","tags":["5-8个话题，不带#"]}}"""
    return chat_json(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=4000,
    )


# ---------------------------------------------------------------- 5. 改写
def revise(note: dict, hits: list[dict]):
    data = load_assets()
    draft = f"{note['title']}\n\n{note['body']}"
    result = audit_mod.audit(draft)
    if not result["hits"]:
        return note, {"attempts": 0, "fixed": [], "audit": result}

    problems = "\n".join(
        f"- 「{h['span']}」→ {h['rule_name']}（{h['basis']}）；上下文：…{h['context']}…"
        for h in result["hits"]
    )
    prompt = f"""这篇笔记违反了广告法和平台规范，请改到合规，同时保留原有的信息量和人味。只输出 JSON。

原文：
{draft}

问题清单：
{problems}

改写原则：
- 只改有问题的句子，不要重写整篇
- 违规主张直接换成可验证的体感描述，不要换个词继续说同样的话
- 允许承认局限（"用久了还是会干"也算合规且更可信）
- 改完仍要满足：第一人称、有具体细节、不出现禁止词

输出：{{"title":"","body":"","changes":["把X改成了Y，因为Z"]}}"""
    new = chat_json(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=4000,
    )
    revised = {
        "title": new.get("title") or note["title"],
        "body": new.get("body") or note["body"],
        "tags": note["tags"],
    }
    changes = list(new.get("changes", []))
    after = audit_mod.audit(f"{revised['title']}\n\n{revised['body']}")

    # 改完还有违规就再来一轮：模型经常只改字面，不改主张。
    if not after["passed"]:
        second = chat_json(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"""上一轮改写没改干净。这次必须让所有违规说法消失，不要只换同义词。

仍有问题：
{chr(10).join('- 「' + h['span'] + '」→ ' + h['rule_name'] for h in after['hits'])}

原文：
{revised['title']}

{revised['body']}

只输出 JSON：{{"title":"","body":"","changes":["改了什么"]}}"""},
            ],
            temperature=0.4,
            max_tokens=4000,
        )
        if second.get("body"):
            revised = {
                "title": second.get("title") or revised["title"],
                "body": second["body"],
                "tags": revised["tags"],
            }
            changes += list(second.get("changes", []))
            after = audit_mod.audit(f"{revised['title']}\n\n{revised['body']}")

    return revised, {
        "attempts": 2 if changes and not after["passed"] else 1,
        "fixed": changes,
        "before": result,
        "audit": after,
    }


# ---------------------------------------------------------------- 6. 度量
def measure(note: dict, hits: list[dict], structure: str):
    text = f"{note['title']}\n{note['body']}"
    data = load_assets()
    voice_issues = audit_mod.style_check(note["body"], data["brand"]["voice"])
    all_points = [
        {"id": x["id"], "title": x["title"], "tags": x.get("tags", [])}
        for x in data["assets"]
        if x["type"] == "卖点"
    ]
    used, missed = audit_mod.coverage(text, all_points)
    paras = [p for p in note["body"].split("\n") if p.strip()]
    return {
        "chars": len(note["body"]),
        "paras": len(paras),
        "avg_para": round(len(note["body"]) / max(len(paras), 1)),
        "title_len": len(note["title"]),
        "tags": len(note["tags"]),
        "points_used": [u["id"] for u in used],
        "points_missed": [m["id"] for m in missed],
        "coverage": f"{len(used)}/{len(used) + len(missed)}",
        "voice_issues": voice_issues,
        "structure": structure,
        "facts": factcheck.check(note["body"]),
    }


# ------------------------------------------------------- 0. 对照组：直接让模型写
def baseline(brief: dict):
    """没有品牌资产、没有调性、没有闸门的写法——这是大多数团队现在的默认做法。

    留这条对照组不是为了证明"AI 不行"，而是为了说明：同一个模型，喂不喂品牌资产、
    过不过闸门，产出差的是量级。没有对照组，谁都可以说自己的系统更好。
    """
    demands = brief.get("client_demands") or []
    prompt = f"""你是小红书爆款文案专家。请为「澄野氨基酸沐浴油」写一篇小红书种草笔记。

要求：有网感、够爆、能带货，标题要抓眼球，多用种草语气。
目标人群：{brief.get('audience', '秋冬干皮女生')}
痛点：{brief.get('pain', '洗完澡皮肤干')}
客户明确要求（必须原样出现在标题或正文里，一个字都不要替客户改）：{('；'.join(demands)) if demands else '无'}

只输出 JSON：{{"title":"","body":"","tags":[]}}"""
    draft = chat_json(
        [{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=3000,
    )
    draft["tags"] = draft.get("tags") or []
    text = f"{draft.get('title','')}\n{draft.get('body','')}"
    a = audit_mod.audit(text)
    used, missed = audit_mod.coverage(
        text,
        [
            {"id": x["id"], "title": x["title"], "tags": x.get("tags", [])}
            for x in load_assets()["assets"]
            if x["type"] == "卖点"
        ],
    )
    return {
        "title": draft.get("title", ""),
        "body": draft.get("body", ""),
        "tags": draft["tags"],
        "audit": a,
        "coverage": f"{len(used)}/{len(used) + len(missed)}",
        "points_used": [u["id"] for u in used],
        "voice_issues": audit_mod.style_check(draft.get("body", ""), []),
        "facts": factcheck.check(draft.get("body", "")),
        "chars": len(draft.get("body", "")),
    }


# ---------------------------------------------------------------- 主流程
def run(brief_text: str, on_step=None):
    log = []

    def emit(name, detail, sec=None):
        item = {"step": name, "detail": detail}
        if sec is not None:
            item["sec"] = sec
        log.append(item)
        if on_step:
            on_step(item)

    t0 = time.time()
    brief, s = _step("brief", parse_brief, brief_text)
    emit("brief", f"识别到 {len(brief.get('key_points', []))} 个卖点、{len(brief.get('avoid', []))} 条禁区", s)

    (queries, hits), s = _step("rag", rag, brief)
    emit("rag", f"{len(queries)} 条检索式 → 召回 {len(hits)} 条品牌资产", s)

    strat, s = _step("strategy", strategize, brief, hits)
    strategies = strat.get("strategies", [])[:3]
    emit("strategy", " / ".join(x.get("structure", "") for x in strategies), s)

    with ThreadPoolExecutor(max_workers=4) as pool:
        base_future = pool.submit(baseline, brief)
        drafts = list(pool.map(lambda st: write_one(st, brief, hits), strategies))
        base = base_future.result()
    emit("write", f"{len(drafts)} 篇初稿 + 1 篇对照组", round(time.time() - t0, 1))


    notes = []
    for strategy, draft in zip(strategies, drafts):
        draft["tags"] = draft.get("tags") or []
        # 统一用 "标题\n\n正文" 作为审计文本，前端才能按位置高亮
        pre = audit_mod.audit(f"{draft['title']}\n\n{draft['body']}")
        final, rev = revise(draft, hits)
        post = audit_mod.audit(f"{final['title']}\n\n{final['body']}")
        notes.append(
            {
                **strategy,
                "title": final["title"],
                "body": final["body"],
                "tags": final["tags"],
                "audit_before": pre,
                "audit": post,
                "changes": rev.get("fixed", []),
                "body_offset": len(final["title"]) + 2,
                "metrics": measure(final, hits, strategy.get("structure", "")),
            }
        )

    ok = sum(1 for n in notes if n["audit"]["passed"])
    total_sec = round(time.time() - t0, 1)
    avg_cov = sum(
        int(n["metrics"]["coverage"].split("/")[0]) for n in notes
    ) / max(len(notes), 1)
    total_points = int(notes[0]["metrics"]["coverage"].split("/")[1]) if notes else 0
    emit("audit", f"合规一次通过 {ok}/{len(notes)}，耗时 {total_sec}s", total_sec)

    return {
        "brief": brief,
        "retrieval": {"queries": queries, "hits": hits},
        "notes": notes,
        "baseline": base,
        "compare": {
            "with_system": {
                "violations": sum(1 for n in notes if n["audit"]["violation_count"]),
                "violation_items": sum(n["audit"]["violation_count"] for n in notes),
                "fact_items": sum(n["metrics"]["facts"]["count"] for n in notes),
                "coverage": f"{avg_cov:.1f}/{total_points}",
                "notes": len(notes),
            },
            "baseline": {
                "violations": 1 if base["audit"]["violation_count"] else 0,
                "violation_items": base["audit"]["violation_count"],
                "fact_items": base["facts"]["count"],
                "coverage": base["coverage"],
                "notes": 1,
            },
        },
        "summary": {
            "notes": len(notes),
            "compliance_pass": f"{ok}/{len(notes)}",
            "total_sec": total_sec,
            "llm_calls": 1 + 1 + 1 + len(notes) + sum(
                1 for n in notes if n["audit_before"]["total"] > 0
            ),
        },
        "trace": log,
    }


if __name__ == "__main__":
    brief = "澄野氨基酸沐浴油，秋冬推一波。想打北方干皮、洗完腿上起白屑的人。主推三油配方和无香精版，别碰可卸防晒那条因为没报告。要有用过的真实感，不要一看就是广告。"
    out = run(brief, on_step=lambda i: print("  [step]", i["step"], "-", i["detail"]))
    for n in out["notes"]:
        print("\n===", n["structure"], "|", n["title"])
        print(n["body"])
        print("tags:", n["tags"], "| metrics:", n["metrics"])
        print("changes:", n["changes"])
