"""品牌事实核对：抓模型编造的数字。

营销场景里最贵的错误不是违禁词，是**编造事实**——价格、成分、检测结论、使用时长。
违禁词会让稿子被删，编造事实会让品牌方被投诉、被平台处罚，而且往往在投放后才发现。

做法：把文案里的量化断言抽出来，和品牌资产里的数字白名单比对，命不中就是「无据」。
为什么不用 LLM 判断：LLM 自己就是编造的一方，让它自查等于让当事人自己作证。
"""
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "brand_assets.json"

# 只核对"商业事实类"数字：价格、百分比、倍数、检测周期、规格。
# 体验类数字（39度水温、一泵半、用了一个月）属于真人细节，不该被当成事实错误。
FACT_PATTERNS = [
    (r"(\d+(?:\.\d+)?)\s*(?:元|块|RMB|rmb)", "价格"),
    (r"[¥￥]\s*(\d+(?:\.\d+)?)", "价格"),
    (r"(\d+(?:\.\d+)?)\s*%", "百分比"),
    (r"(\d+(?:\.\d+)?)\s*倍", "倍数"),
    (r"(\d+)\s*(?:天|日)\s*(?:见效|起效|改善|搞定)", "时效承诺"),
    (r"(\d+)\s*(?:ml|毫升|g|克)", "规格"),
]


def _numbers(text: str):
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


# 白名单只收「带单位的数字」。直接把整份 JSON 扫一遍数字会把规则表里的
# 「7天」「100%」也算进白名单，结果模型写「7天见效」反过来被判合规——这是
# 我踩过的坑：白名单的污染会让闸门静默失效。
UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:元|块|ml|毫升|mL|天|日|%|倍|g|克)")
PH_RE = re.compile(r"pH\s*(\d+(?:\.\d+)?)")


def whitelist():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    # 只取「事实性字段」。品牌红线里也写着「7天见效」「100%」这类反面例子，
    # 把它们扫进白名单等于给违规说法发通行证。
    brand = data["brand"]
    pool = json.dumps(
        {
            "meta": [brand.get("name"), brand.get("product"), brand.get("positioning"), brand.get("target")],
            "assets": data["assets"],
        },
        ensure_ascii=False,
    )
    return {m.group(1) for m in UNIT_RE.finditer(pool)} | {
        m.group(1) for m in PH_RE.finditer(pool)
    }


# 百分比只有在功效语境里才是"事实宣称"。湿度 31%、含水量 40% 这种观测值不该被拦，
# 否则规则就成了噪音源——一个会被无视的检查比没有检查更危险。
CLAIM_WORDS = ["提升", "改善", "减少", "降低", "提高", "有效", "见效", "修复", "淡", "退", "增", "达"]


def _in_claim_context(text, start, end):
    window = text[max(0, start - 24) : end + 24]
    return any(w in window for w in CLAIM_WORDS)


def check(text: str):
    ok = whitelist()
    flagged = []
    for pattern, kind in FACT_PATTERNS:
        for m in re.finditer(pattern, text):
            value = m.group(1)
            if kind == "规格":
                continue  # 规格只出现在产品页描述里，正文很少用，不做拦截
            if kind == "百分比" and not _in_claim_context(text, m.start(), m.end()):
                continue
            if value in ok:
                continue
            flagged.append(
                {
                    "kind": kind,
                    "span": m.group(0),
                    "value": value,
                    "context": text[max(0, m.start() - 16) : m.end() + 10],
                    "reason": f"品牌资产里没有 {value} 这个数（{kind}），属于无据引用",
                }
            )
    return {
        "checked": len(re.findall(r"\d+", text or "")),
        "flagged": flagged,
        "count": len(flagged),
        "passed": len(flagged) == 0,
    }


if __name__ == "__main__":
    print(check("澄野氨基酸沐浴油398元买了两瓶，我坚持用了7天见效，屏障修复率98%，一瓶380ml。"))
