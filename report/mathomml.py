# -*- coding: utf-8 -*-
"""把一小部分 LaTeX 语法转成 Word 公式对象（OMML）。

只实现本报告用到的语法：希腊字母与常用运算符、上下标、\\frac、\\sum/\\prod、
\\left( \\right)、cases 环境、\\mathbf/\\boldsymbol（粗斜体向量）、\\mathrm/\\text、
\\hat/\\bar 以及若干空白命令。转换结果是 m:oMath 元素，可以直接放进段落。
"""

import re

from docx.oxml import OxmlElement
from docx.oxml.ns import qn

MATH_FONT = "Cambria Math"
TEXT_FONT = "宋体"

# 数学符号：命令 -> 字符
SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ",
    "varepsilon": "ε", "theta": "θ", "lambda": "λ", "mu": "μ", "pi": "π",
    "sigma": "σ", "Sigma": "Σ", "tau": "τ", "phi": "φ", "varphi": "φ",
    "omega": "ω", "Omega": "Ω", "eta": "η", "rho": "ρ",
    "ell": "ℓ", "nabla": "∇", "partial": "∂", "top": "⊤",
    "in": "∈", "notin": "∉", "ge": "≥", "geq": "≥", "le": "≤", "leq": "≤",
    "ne": "≠", "neq": "≠", "approx": "≈", "equiv": "≡",
    "cdot": "·", "times": "×", "div": "÷", "mid": "|", "vert": "|",
    "sum": "∑", "prod": "∏", "int": "∫",
    "infty": "∞", "dots": "…", "ldots": "…", "cdots": "⋯",
    "to": "→", "rightarrow": "→", "leftarrow": "←", "gets": "←", "Rightarrow": "⇒", "Leftrightarrow": "⇔",
    "sim": "∼", "propto": "∝", "pm": "±", "mp": "∓",
    "lbrace": "{", "rbrace": "}", "langle": "⟨", "rangle": "⟩",
    "quad": "\u00a0\u00a0", "qquad": "\u00a0\u00a0\u00a0\u00a0",
    ",": "\u2009", ";": "\u2005", ":": "\u2005", " ": "\u00a0",
    "!": "", "{": "{", "}": "}", "%": "%", "#": "#", "&": "&", "_": "_",
}

# 直立显示的函数名
FUNCTIONS = {"ln", "log", "exp", "max", "min", "sgn", "arg"}

# 直立显示的常用名字（\mathrm 之外的约定俗成写法）
UPRIGHT_WORDS = {"accuracy", "Likelihood"}

GREEK_UPPER = set("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")


def _el(tag, *children, **attrs):
    el = OxmlElement("m:" + tag)
    for key, val in attrs.items():
        el.set(qn("m:" + key), val)
    for child in children:
        if child is None:
            continue
        if isinstance(child, (list, tuple)):
            for item in child:
                el.append(item)
        else:
            el.append(child)
    return el


def _run(text, style="i", size_half_points=22):
    """style: i 斜体 / p 直立 / b 粗体 / bi 粗斜体。"""
    run = _el("r")
    rpr = _el("rPr")
    if style == "p":
        # 直立文本用 m:nor，Word 与 LibreOffice 都会渲染为正体
        rpr.append(_el("nor"))
    else:
        rpr.append(_el("sty", val=style))
    run.append(rpr)
    wrpr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), MATH_FONT)
    fonts.set(qn("w:hAnsi"), MATH_FONT)
    fonts.set(qn("w:eastAsia"), TEXT_FONT)
    wrpr.append(fonts)
    if style in ("b", "bi"):
        wrpr.append(OxmlElement("w:b"))
        wrpr.append(OxmlElement("w:bCs"))
    if style in ("i", "bi"):
        wrpr.append(OxmlElement("w:i"))
        wrpr.append(OxmlElement("w:iCs"))
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(size_half_points))
    wrpr.append(sz)
    run.append(wrpr)
    t = _el("t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    run.append(t)
    return run


_RPR_ORDER = [
    "w:rStyle", "w:rFonts", "w:b", "w:bCs", "w:i", "w:iCs",
    "w:caps", "w:smallCaps", "w:strike", "w:dstrike", "w:outline",
    "w:shadow", "w:emboss", "w:imprint", "w:noProof", "w:snapToGrid",
    "w:vanish", "w:webHidden", "w:color", "w:spacing", "w:w", "w:kern",
    "w:position", "w:sz", "w:szCs", "w:highlight", "w:u", "w:effect",
    "w:bdr", "w:shd", "w:fitText", "w:vertAlign", "w:rtl", "w:cs",
    "w:em", "w:lang", "w:eastAsianLayout", "w:specVanish", "w:oMath",
]


def _reorder_rpr(wrpr):
    """按 OOXML 规定的顺序重排 w:rPr 子元素，避免 Word 报错。"""
    children = list(wrpr)
    def key(el):
        tag = el.tag.split("}")[-1]
        name = "w:" + tag
        return _RPR_ORDER.index(name) if name in _RPR_ORDER else len(_RPR_ORDER)
    for child in children:
        wrpr.remove(child)
    for child in sorted(children, key=key):
        wrpr.append(child)


def _e(tag, items):
    element = _el(tag)
    for item in items:
        element.append(item)
    return element


def _frac(num, den):
    return _el("f", _el("fPr"), _e("num", num), _e("den", den))


def _sub(base, sub):
    return _el("sSub", _el("sSubPr"), _e("e", base), _e("sub", sub))


def _sup(base, sup):
    return _el("sSup", _el("sSupPr"), _e("e", base), _e("sup", sup))


def _subsup(base, sub, sup):
    return _el("sSubSup", _el("sSubSupPr"), _e("e", base), _e("sub", sub), _e("sup", sup))


def _nary(char, sub, sup, body):
    pr = _el("naryPr")
    pr.append(_el("chr", val=char))
    pr.append(_el("limLoc", val="undOvr"))
    pr.append(_el("subHide", val="0" if sub else "1"))
    pr.append(_el("supHide", val="0" if sup else "1"))
    return _el(
        "nary", pr,
        _e("sub", sub or []),
        _e("sup", sup or []),
        _e("e", body),
    )


def _delim(items, begin="(", end=")"):
    pr = _el("dPr")
    pr.append(_el("begChr", val=begin))
    pr.append(_el("endChr", val=end))
    return _el("d", pr, _e("e", items))


def _cases(rows):
    arr = _el("eqArr")
    for row in rows:
        arr.append(_e("e", row))
    pr = _el("dPr")
    pr.append(_el("begChr", val="{"))
    pr.append(_el("endChr", val=""))
    return _el("d", pr, _e("e", [arr]))


class _Parser:
    def __init__(self, source, size=22):
        self.s = source
        self.i = 0
        self.size = size

    # -- 基础 -------------------------------------------------------------
    def peek(self, n=1):
        return self.s[self.i:self.i + n]

    def eof(self):
        return self.i >= len(self.s)

    def skip_ws(self):
        while not self.eof() and self.s[self.i] in " \t\n":
            self.i += 1

    def read_group(self):
        """读取 {…} 并返回内部元素列表。"""
        self.skip_ws()
        if self.peek() != "{":
            return self.parse_atom()
        self.i += 1
        items = self.parse_sequence(stop="}")
        if self.peek() == "}":
            self.i += 1
        return items

    # -- 解析 -------------------------------------------------------------
    def parse_atom(self):
        """返回一个原子的元素列表。"""
        self.skip_ws()
        if self.eof():
            return []
        ch = self.s[self.i]
        if ch == "{":
            self.i += 1
            items = self.parse_sequence(stop="}")
            if self.peek() == "}":
                self.i += 1
            return items
        if ch == "\\":
            return self.parse_command()
        if ch == "}":
            return []
        self.i += 1
        return [_run(ch, self.style_of(ch), self.size)]

    @staticmethod
    def style_of(ch):
        if ch.isalpha() or ch in GREEK_UPPER:
            return "i"
        return "p"

    def parse_command(self):
        self.i += 1  # 跳过反斜杠
        start = self.i
        while not self.eof() and (self.s[self.i].isalpha()):
            self.i += 1
        name = self.s[start:self.i]
        if not name:  # 反斜杠 + 单字符，如 \{ \, \%
            name = self.s[self.i:self.i + 1]
            self.i += 1

        if name == "frac":
            num = self.read_group()
            den = self.read_group()
            return [_frac(num, den)]
        if name in ("mathbf", "boldsymbol", "bm"):
            group = self.read_group()
            style = "b" if name == "mathbf" else "bi"
            return self._restyle(group, style)
        if name == "mathrm":
            group = self.read_group()
            return self._restyle(group, "p")
        if name == "text":
            raw = self._raw_group()
            return [_run(raw, "p", self.size)]
        if name in FUNCTIONS:
            return [_run(name, "p", self.size)]
        if name in ("sum", "prod"):
            return [self._nary_of(name)]
        if name == "left":
            begin = self.s[self.i]
            self.i += 1
            body = self.parse_sequence(stop="\\right")
            end = ")"
            if self.s[self.i:self.i + 6] == "\\right":
                self.i += 6
                end = self.s[self.i:self.i + 1]
                self.i += 1
            if begin == ".":
                begin = ""
            if end == ".":
                end = ""
            return [_delim(body, begin, end)]
        if name == "begin" and self.s[self.i:self.i + 6] == "{cases":
            return [self._cases_env()]
        if name == "mathbb":
            raw = self._raw_group()
            mapping = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ"}
            return [_run(mapping.get(raw, raw), "p", self.size)]
        if name == "mathcal":
            raw = self._raw_group()
            mapping = {"L": "ℒ", "M": "ℳ", "N": "𝒩", "R": "ℛ"}
            return [_run(mapping.get(raw, raw), "p", self.size)]
        if name == "hat":
            group = self.read_group()
            text = "".join(t.text or "" for t in self._collect_t(group))
            return [_run(text + "\u0302", "i", self.size)]
        if name == "bar":
            group = self.read_group()
            text = "".join(t.text or "" for t in self._collect_t(group))
            return [_run(text + "\u0304", "i", self.size)]
        if name in SYMBOLS:
            text = SYMBOLS[name]
            style = "p"
            if text.isalpha() and len(text) == 1 and text not in "ΣΔΩ":
                style = "i"
            return [_run(text, style, self.size)] if text else []
        # 未知命令：按直立文本输出命令名，避免整体失败
        return [_run("\\" + name, "p", self.size)]

    def _nary_of(self, name):
        char = "∑" if name == "sum" else "∏"
        sub, sup = [], []
        # 依次吃掉上标和下标（顺序任意）
        for _ in range(2):
            self.skip_ws()
            if self.peek() == "_":
                self.i += 1
                sub = self.read_group()
            elif self.peek() == "^":
                self.i += 1
                sup = self.read_group()
        body = self.parse_sequence(stop=None)
        return _nary(char, sub, sup, body)

    def _cases_env(self):
        # 跳过 \begin{cases}
        end = self.s.find("}", self.i)
        self.i = end + 1
        close = self.s.find("\\end{cases}", self.i)
        if close < 0:
            close = len(self.s)
        body = self.s[self.i:close]
        self.i = min(close + 11, len(self.s))
        rows = []
        for raw_row in body.split("\\\\"):
            if not raw_row.strip():
                continue
            cells = raw_row.split("&")
            row = []
            for idx, cell in enumerate(cells):
                if idx:
                    row.append(_run("\u00a0\u00a0\u00a0", "p", self.size))
                row.extend(_Parser(cell, self.size).parse_sequence())
            if row:
                rows.append(row)
        return _cases(rows)

    def _raw_group(self):
        self.skip_ws()
        if self.peek() != "{":
            ch = self.s[self.i:self.i + 1]
            self.i += 1
            return ch
        self.i += 1
        depth, start = 1, self.i
        while not self.eof() and depth:
            if self.s[self.i] == "{":
                depth += 1
            elif self.s[self.i] == "}":
                depth -= 1
                if depth == 0:
                    break
            self.i += 1
        text = self.s[start:self.i]
        self.i += 1
        return text

    @staticmethod
    def _restyle(items, style):
        for item in items:
            for run in item.iter(qn("m:r")):
                rpr = run.find(qn("m:rPr"))
                if rpr is not None:
                    sty = rpr.find(qn("m:sty"))
                    if sty is not None:
                        sty.set(qn("m:val"), style)
                    else:
                        nor = rpr.find(qn("m:nor"))
                        if nor is not None and style in ("b", "bi"):
                            rpr.remove(nor)
                            rpr.append(_el("sty", val=style))
                wrpr = run.find(qn("w:rPr"))
                if wrpr is not None and style in ("b", "bi"):
                    if wrpr.find(qn("w:b")) is None:
                        wrpr.insert(0, OxmlElement("w:b"))
                    if style == "bi" and wrpr.find(qn("w:i")) is None:
                        wrpr.insert(0, OxmlElement("w:i"))
                    _reorder_rpr(wrpr)
        return items

    @staticmethod
    def _collect_t(items):
        out = []
        for item in items:
            out.extend(item.iter(qn("m:t")))
        return out

    def parse_sequence(self, stop=None):
        items = []
        while not self.eof():
            if stop == "}" and self.peek() == "}":
                break
            if stop == "\\right" and self.s[self.i:self.i + 6] == "\\right":
                break
            if stop and stop != "}" and stop != "\\right" and self.s.startswith(stop, self.i):
                break
            if self.peek() in "^_":
                mark = self.s[self.i]
                self.i += 1
                script = self.read_group()
                base = items[-1:] if items else []
                if base:
                    items = items[:-1]
                head = base[0] if base else _run("", "p", self.size)
                if mark == "^":
                    if head.tag == qn("m:sSub"):
                        # 已有下标，合并成 sSubSup
                        old_sub = head.find(qn("m:sub"))
                        old_base = head.find(qn("m:e"))
                        merged = _subsup(
                            [c for c in old_base],
                            [c for c in old_sub] if old_sub is not None else [],
                            script,
                        )
                        items.append(merged)
                    else:
                        items.append(_sup([head], script))
                else:
                    if head.tag == qn("m:sSup"):
                        old_sup = head.find(qn("m:sup"))
                        old_base = head.find(qn("m:e"))
                        merged = _subsup(
                            [c for c in old_base],
                            script,
                            [c for c in old_sup] if old_sup is not None else [],
                        )
                        items.append(merged)
                    else:
                        items.append(_sub([head], script))
                continue
            atom = self.parse_atom()
            items.extend(atom)
        return _merge_run(items)


def _merge_run(items):
    """合并相邻且样式相同的 m:r，减少节点数。"""
    out = []
    for item in items:
        if item.tag != qn("m:r") or not out or out[-1].tag != qn("m:r"):
            out.append(item)
            continue
        prev = out[-1]
        prev_sty = prev.find(qn("m:rPr") + "/" + qn("m:sty"))
        cur_sty = item.find(qn("m:rPr") + "/" + qn("m:sty"))
        prev_val = prev_sty.get(qn("m:val")) if prev_sty is not None else None
        cur_val = cur_sty.get(qn("m:val")) if cur_sty is not None else None
        if prev_val != cur_val:
            out.append(item)
            continue
        prev_text = prev.find(qn("m:t"))
        cur_text = item.find(qn("m:t"))
        if prev_text is not None and cur_text is not None:
            prev_text.text = (prev_text.text or "") + (cur_text.text or "")
        else:
            out.append(item)
    return out


def omath(latex, size=22):
    """把一段 LaTeX 子集转成 m:oMath 元素；size 为半磅字号（22 表示 11pt）。"""
    parser = _Parser(latex.strip(), size)
    body = parser.parse_sequence()
    return _el("oMath", body)
