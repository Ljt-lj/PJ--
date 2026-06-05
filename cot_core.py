"""Shared CoT prompt, extraction, and answer-matching utilities."""

from __future__ import annotations

import re
from collections import Counter

# Best validated on dev-50 (seed=42, Qwen2.5-0.5B via Ollama):
#   prompt_mode=compact, temperature=0.0, top_p=1.0  ->  36% accuracy
BEST_PROMPT_MODE = "compact"
BEST_TEMPERATURE = 0.0
BEST_TOP_P = 1.0

BASE_FEW_SHOT = """\
示例1（四则运算）：
问题：商店有4框苹果，每框55千克，已经卖出135千克，还剩多少千克苹果?
推理：总重量 = 4 × 55 = 220 千克；剩余 = 220 - 135 = 85 千克。
答案：85

示例2（多步运算）：
问题：玩具厂生产了960个电子玩具，每3个装一盒，每5盒装一箱，一共装了多少箱?
推理：盒数 = 960 ÷ 3 = 320 盒；箱数 = 320 ÷ 5 = 64 箱。
答案：64"""

TAG_FEW_SHOT: dict[str, str] = {
    "fraction": """\
示例（分数）：
问题：把48平均分成6份，取其中5份，占总数的几分之几?
推理：每份 = 48 ÷ 6 = 8；取5份 = 40；占总数 = 40/48 = 5/6。
答案：5/6""",
    "decimal": """\
示例（小数）：
问题：91.64与7.36的和乘43.6与3.6的差，积是多少？
推理：(91.64 + 7.36) × (43.6 - 3.6) = 99 × 40 = 3960
答案：3960""",
    "at_least": """\
示例（至少/进一法）：
问题：两个老师带着30名同学在公园里划船，每条船最多坐3人，至少需多少条船?
推理：总人数 = 2 + 30 = 32 人；32 ÷ 3 = 10 余 2，有余数需多一条船，共 11 条。
答案：11""",
    "at_most": """\
示例（至多/去尾法）：
问题：每盒蛋糕7.90元，50元最多可以买多少盒蛋糕?
推理：50 ÷ 7.90 ≈ 6.32，最多买整盒，取整数部分 6 盒。
答案：6""",
    "average": """\
示例（平均数）：
问题：小明前3天每天读12页，后4天每天读15页，平均每天读多少页?
推理：总页数 = 3×12 + 4×15 = 36 + 60 = 96；平均 = 96 ÷ 7 ≈ 13.714... 
答案：13.714286""",
    "ratio": """\
示例（和差倍）：
问题：书架上有两层书，共164本，如果从下层取出9本放到上层去，两层书的本书就相同，原来下层比上层多多少本书?
推理：移9本后相等，差为18本；上层=(164-18)/2=73，下层=91；原来下层比上层多 91-73=18 本。
答案：18""",
    "unit_convert": """\
示例（单位换算）：
问题：在一节数学课上，老师讲课用了1/5小时，小组合作用了2/15小时，其余时间做练习，一节课35分钟，学生做练习用了多少小时?
推理：35分钟 = 35/60 = 7/12 小时；已用 = 1/5 + 2/15 = 3/15 + 2/15 = 5/15 = 1/3 小时；练习 = 7/12 - 1/3 = 7/12 - 4/12 = 3/12 = 1/4 小时。
答案：1/4""",
    "remainder": """\
示例（剩余）：
问题：食堂买来105千克的萝卜，已经吃了35千克，还剩多少千克?
推理：剩余 = 105 - 35 = 70 千克。
答案：70""",
}

TAG_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("fraction", re.compile(r"分之|[\d]+/[\d]+|分数")),
    ("decimal", re.compile(r"\d+\.\d+")),
    ("at_least", re.compile(r"至少|最少|不少于")),
    ("at_most", re.compile(r"至多|最多|不超过|最多能|最多可")),
    ("average", re.compile(r"平均|均值")),
    ("ratio", re.compile(r"比.*(?:多|少|是)|倍数|倍")),
    ("unit_convert", re.compile(r"千米|厘米|毫米|小时|分钟|吨|千克|克|升|毫升|元|角|分")),
    ("remainder", re.compile(r"余|剩下|剩余|还剩")),
]

RULES_TEXT = """\
解题规则：
1. 仔细读题，提取所有已知数和所求量。
2. 「至少/最少」用进一法（有余数则+1）；「至多/最多」用去尾法（取整数部分）。
3. 分数答案若未约分，按题目要求输出分数（如 1/4、4/5）或小数，与计算结果一致即可。
4. 单位换算注意：1小时=60分钟，1千米=1000米，1吨=1000千克。
5. 最后只输出数字、分数或百分数（如 18.8 或 18.8%），不要带单位名称。
6. 务必在最后一行输出「答案：xxx」，xxx 仅为最终数值。"""

USER_TEMPLATE = "问题：{question}\n请逐步推理，最后一行严格按「答案：<数字或分数>」格式输出，不要添加其他内容。"

USER_TEMPLATE_COMPACT = "问题：{question}\n分步计算，最后一行只写：答案：<数字或分数>"

COMPACT_FEW_SHOT = """\
示例：
问题：商店有4框苹果，每框55千克，已经卖出135千克，还剩多少千克?
计算：4×55=220，220-135=85
答案：85"""

ANSWER_LINE_PATTERN = re.compile(
    r"^[\s*\-]*(?:答案|最终答案|答)[：:\s]*([+-]?\d+(?:\.\d+)?(?:/\d+)?%?)\s*[\s。.*]*$",
    re.MULTILINE,
)
ANSWER_INLINE_PATTERNS = [
    re.compile(r"答案[是为：:]\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?%?)"),
    re.compile(r"最终答案[是为：:]\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?%?)"),
    re.compile(r"\\boxed\{([+-]?\d+(?:\.\d+)?(?:/\d+)?%?)\}"),
    re.compile(r"计算[：:].*?=\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?%?)"),
]


def normalize_question(question: str | list) -> str:
    if isinstance(question, list):
        return "".join(str(part) for part in question)
    return str(question).strip()


def classify_question(question: str | list) -> list[str]:
    text = normalize_question(question)
    tags = [name for name, pattern in TAG_RULES if pattern.search(text)]
    return tags or ["general"]


def build_few_shot(question: str | list, max_examples: int = 4) -> str:
    tags = classify_question(question)
    parts = [BASE_FEW_SHOT]
    used = 1
    for tag in tags:
        if tag in TAG_FEW_SHOT and used < max_examples:
            parts.append(TAG_FEW_SHOT[tag])
            used += 1
    return "\n\n".join(parts)


def build_system_prompt(question: str | list, *, compact: bool = False) -> str:
    if compact:
        return build_compact_prompt(question)

    few_shot = build_few_shot(question)
    extra = ""
    tags = classify_question(question)
    if "at_least" in tags:
        extra = "\n注意：本题含「至少/最少」，有余数时需向上取整（进一法）。"
    elif "at_most" in tags:
        extra = "\n注意：本题含「至多/最多」，只能取整数部分（去尾法）。"
    elif "fraction" in tags:
        extra = "\n注意：本题涉及分数，答案可写分数形式（如 3/5）或等价小数。"

    return f"""你是小学数学1-6年级应用题解题助手。请逐步推理，最后给出纯数字或分数答案（不含单位）。

{RULES_TEXT}{extra}

参考示例：
{few_shot}

输出格式：
推理：<分步计算>
答案：<仅数字或分数>"""


def build_compact_prompt(question: str | list) -> str:
    """Compact CoT prompt — best profile on dev-50 (36% with greedy decoding)."""
    tags = classify_question(question)
    hints: list[str] = []
    if "at_least" in tags:
        hints.append("「至少/最少」有余数则+1")
    if "at_most" in tags:
        hints.append("「至多/最多」只取整数部分")
    if "fraction" in tags:
        hints.append("分数题答案写分数如3/5或小数")
    if "unit_convert" in tags:
        hints.append("注意单位：1小时=60分钟，1吨=1000千克")
    hint_text = f"\n提示：{'；'.join(hints)}。" if hints else ""

    return f"""你是小学数学应用题助手。先列算式再给出最终数字答案（不带单位）。{hint_text}

{COMPACT_FEW_SHOT}

格式：
计算：<算式步骤>
答案：<仅数字或分数>"""


DIRECT_INSTRUCTION = (
    "这是小学数学1-6年级的校内题目，无需进行分析，请直接输出数字答案，不带单位。"
)


def build_direct_messages(question: str | list, instruction: str = DIRECT_INSTRUCTION) -> tuple[str, str]:
    q = normalize_question(question)
    system = instruction
    user = q
    return system, user


def get_user_prompt(question: str | list, *, compact: bool = False) -> str:
    q = normalize_question(question)
    template = USER_TEMPLATE_COMPACT if compact else USER_TEMPLATE
    return template.format(question=q)


def normalize_number(value: str) -> str:
    value = str(value).strip().replace("，", "").replace(",", "").replace("％", "%")
    if not value:
        return "0"
    if value.endswith("%"):
        value = value[:-1].strip()
    if "/" in value:
        parts = value.split("/")
        if len(parts) == 2:
            try:
                num = float(parts[0]) / float(parts[1])
                if abs(num - round(num)) < 1e-9:
                    return str(int(round(num)))
                return str(round(num, 6)).rstrip("0").rstrip(".")
            except (ValueError, ZeroDivisionError):
                return value
    try:
        num = float(value)
        if abs(num - round(num)) < 1e-9:
            return str(int(round(num)))
        return str(round(num, 6)).rstrip("0").rstrip(".")
    except ValueError:
        return value


def to_float(value: str) -> float | None:
    value = str(value).strip()
    if "/" in value:
        parts = value.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return None
    try:
        return float(value)
    except ValueError:
        return None


def answers_equal(pred: str, gold: str, tol: float = 1e-4) -> bool:
    p, g = str(pred).strip(), str(gold).strip()
    if p == g:
        return True
    pn, gn = normalize_number(p), normalize_number(g)
    if pn == gn:
        return True
    pf, gf = to_float(p), to_float(g)
    if pf is not None and gf is not None:
        return abs(pf - gf) <= tol
    return False


def extract_answer(text: str, *, reasoning: str | None = None) -> str:
    """Extract final numeric answer; prefer explicit answer line over fallback."""
    if not text and reasoning:
        text = reasoning
    if not text:
        return "0"

    # Prefer final answer from content; scan line-by-line from bottom
    for block in (text, reasoning or ""):
        lines = block.strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            m = ANSWER_LINE_PATTERN.match(line)
            if m:
                return normalize_number(m.group(1))

        for pattern in ANSWER_INLINE_PATTERNS:
            matches = pattern.findall(block)
            if matches:
                return normalize_number(matches[-1])

    # Fallback: last number in text (avoid numbers from question echo)
    numbers = re.findall(r"(?<![\d./])([+-]?\d+(?:\.\d+)?(?:/\d+)?)", text)
    if numbers:
        return normalize_number(numbers[-1])
    return "0"


def format_submit_answer(question: str | list, answer: str) -> str:
    """Format answer for competition CSV (add % when question asks for percent)."""
    q = normalize_question(question)
    ans = str(answer).strip()
    if re.search(r"百分之|百分比|百分数|利润率|增加了百分之|降低了百分之|增长了百分之", q):
        if not ans.endswith("%"):
            ans = normalize_number(ans)
            return f"{ans}%"
    return normalize_number(ans) if not ans.endswith("%") else ans


def vote_answer(candidates: list[str]) -> str:
    cleaned = [normalize_number(c) for c in candidates if c]
    if not cleaned:
        return "0"
    counts = Counter(cleaned)
    return counts.most_common(1)[0][0]
