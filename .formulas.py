"""Formula parsing and rendering for OmniOpt2.

This module lets users describe a function to optimize in either:

* **Infix** notation (``"sin(a*x) + Sum(i**2, (i, 0, b))"``) — parsed via
  ``sympy.parsing.sympy_parser.parse_expr``.
* **LaTeX** notation (``r"\\sin(a\\,x) + \\sum_{i=0}^{b} i^{2}"``) — parsed via
  ``sympy.parsing.latex.parse_latex`` if the optional ``antlr4-python3-runtime``
  is available, otherwise via a small built-in preprocessor + ``parse_expr``.

From a parsed ``sympy.Expr`` we can:

* Suggest hyperparameter names (the free symbols of the expression).
* Build a numeric call-back via :func:`sympy.lambdify` so OmniOpt can call
  ``f(a=..., b=...)`` directly without any user-written program.
* Render the formula as ASCII (for the shell header) or as LaTeX with
  per-parameter ``\\underbrace`` (for the share viewer).

The functions in this module never touch the network, the filesystem or
``submitit``; they are intentionally pure so they can be unit-tested in
isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from sympy import (
        Expr,
        Float,
        Integer,
        Symbol,
        latex as sympy_latex,
        pretty as sympy_pretty,
    )
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
    from sympy.utilities.lambdify import lambdify
except ImportError as exc:  # pragma: no cover - sympy is a hard requirement
    raise ImportError(
        "sympy is required for OmniOpt formula support. "
        "Install it with `pip install sympy`."
    ) from exc


__all__ = [
    "Formula",
    "SuggestedParameter",
    "parse_infix",
    "parse_latex_or_infix",
    "strip_lhs_assignment",
    "strip_text_macros",
    "preprocess_latex",
    "suggest_hyperparameters",
    "suggest_formula_split",
    "render_ascii",
    "render_latex_with_underbraces",
    "build_lambda",
    "render_latex",
    "choose_latex_or_infix",
]


# ---------------------------------------------------------------------------
# LaTeX preprocessing (so we can parse with parse_expr as a fallback)
# ---------------------------------------------------------------------------

# Macros that are pure formatting and carry no math meaning.  We strip them
# and their braces wholesale.  Each entry maps ``\name`` -> empty.
_TEXT_LIKE_MACROS = (
    "text",
    "textit",
    "textbf",
    "mathrm",
    "textrm",
    "operatorname",
    "mathbf",
    "mathcal",
    "mathbb",
    "mathfrak",
    "mathsf",
    "mathtt",
    "mbox",
    "operatorname*",
    "boldsymbol",
)

# Macros that should be replaced by their sympy equivalent.
_MACRO_REPLACEMENTS = {
    r"\frac": " / ",  # handled by sympy via / already
    r"\dfrac": " / ",
    r"\tfrac": " / ",
    r"\cdot": "*",
    r"\times": "*",
    r"\div": "/",
    r"\left": "",
    r"\right": "",
    r"\, ": " ",
    r"\,": "",
    r"\; ": " ",
    r"\;": "",
    r"\!": "",
    r"\ ": " ",
    r"\quad": "  ",
    r"\qquad": "   ",
    # Bound / infinite notation
    r"\infty": "oo",
    r"\inf": "oo",
    # Limit / "to" notation
    r"\to": ",",
    r"\rightarrow": ",",
    r"\Rightarrow": ",",
}


def strip_text_macros(latex: str) -> str:
    """Remove pure-formatting macros like ``\\text{...}`` or ``\\textit{...}``.

    The content of braces is kept (it may contain math symbols or numbers).
    """
    text = latex
    for macro in _TEXT_LIKE_MACROS:
        # Match \macro{...} with possibly nested braces.
        pattern = re.compile(r"\\" + re.escape(macro) + r"\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
        while True:
            new_text = pattern.sub(r"\1", text)
            if new_text == text:
                break
            text = new_text
    return text


def _sanitize_shell_corruption(text: str) -> str:
    """Repair common shell ``echo`` corruption of LaTeX backslash sequences.

    When a formula is passed through ``echo '...' | base64`` (instead of
    ``printf '%s' '...' | base64``), the shell interprets ``\\f`` as a form
    feed (0x0C), ``\\n`` as newline, ``\\t`` as tab, etc.  This function
    detects and reverses the most common corruptions so the parser still
    produces correct results.
    """
    # \f -> form feed (0x0C):  \frac -> \x0crac, \ff -> \x0cf
    # Repair: form-feed followed by known LaTeX continuations.
    repairs = [
        ("\x0crac", "\\frac"),
        ("\x0cdfrac", "\\dfrac"),
        ("\x0ctfrac", "\\tfrac"),
        ("\x0cbox", "\\fbox"),
        ("\x0c", ""),  # stray form feed with no known continuation
    ]
    for broken, fixed in repairs:
        text = text.replace(broken, fixed)
    # Strip any remaining non-printable control chars (except \t, \n, \r).
    text = "".join(
        ch for ch in text
        if ch in ("\t", "\n", "\r") or ord(ch) >= 32
    )
    return text


def preprocess_latex(latex: str) -> str:
    """Best-effort preprocessing of LaTeX so :func:`parse_expr` can chew it.

    This is intentionally conservative — we only handle the macros that we
    know show up in the wild for OmniOpt users.  Anything that survives is
    passed through verbatim so :func:`parse_expr` (or
    :func:`sympy.parsing.latex.parse_latex`) can complain about it.
    """
    text = _sanitize_shell_corruption(latex)
    text = strip_text_macros(text)

    # \frac{a}{b}  ->  (a)/(b)  (must come before general macro substitution)
    text = _expand_frac(text)

    # \frac{d}{dx} BODY  ->  Derivative(BODY, x)
    text = _expand_derivative_sentinel(text)

    # \sum_{i=0}^{n} f  ->  Sum(f, (i, 0, n))
    text = _expand_sum(text)

    # \prod_{i=0}^{n} f  ->  Product(f, (i, 0, n))
    text = _expand_prod(text)

    # \int_{a}^{b} f \,dx  ->  Integral(f, (x, a, b))
    text = _expand_int(text)

    # \lim_{x \to v} f  ->  Limit(f, x, v)
    text = _expand_lim(text)

    # \sqrt{...}  ->  sqrt(...)
    text = _expand_sqrt(text)

    # |expr|  ->  Abs(expr)
    text = _expand_abs(text)

    # \sin / \cos / \tan / \exp / \log / \ln / \sigma / etc. -> sympy names
    text = re.sub(
        r"\\(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|exp|log|ln|sqrt|abs|sigma|sign|min|max|erf|sigmoid|softmax|relu|leakyrelu|tanh|hardtanh|softplus|gelu|mish|step)\b",
        r"\1",
        text,
    )

    # Drop more LaTeX macros that don't change semantics: spacing,
    # decorative symbols, styling.
    text = re.sub(
        r"\\(hat|tilde|bar|vec|dot|ddot|widehat|widetilde|overbrace|underbrace|overline|underline|cdot|times|ast|circ|bullet|dagger|ddagger|oplus|otimes|equiv|sim|approx|neq|le|ge|leq|geq|to|rightarrow|leftarrow|mapsto|Rightarrow|Leftarrow|partial|nabla|partial)\b",
        r"",
        text,
    )

    # ``\|`` is the LaTeX norm delimiter; we treat it as a plain ``|`` so
    # the absolute-value expander can pair it up correctly.
    text = text.replace(r"\|", "|")

    # Simple macro substitutions
    for macro, replacement in _MACRO_REPLACEMENTS.items():
        text = text.replace(macro, replacement)

    # Drop stray single-token braces that confuse parse_expr (e.g. `{x}^2` -> `x^2`).
    text = re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", r"\1", text)

    # Convert remaining ``{...}`` to ``(...)`` so Python doesn't see them
    # as a set literal.  Only the outer-most non-nested braces are touched
    # (the parser has already resolved all nested braces by this point).
    text = re.sub(r"\{([^{}]*)\}", r"(\1)", text)

    # Convert `^` to `**` for Python's parser.
    text = text.replace("^", "**")

    # Strip any leftover ``{...}`` braces around simple sub-expressions so that
    # things like ``e**{-z}`` become ``e**(-z)`` (parse_expr does not like
    # ``**{...}``).
    text = re.sub(r"\*\*\s*\{([^}]+)\}", r"**(\1)", text)

    # Implicit multiplication: `2x`, `xy`, `)(`, `n(` -> insert `*`.
    text = _add_implicit_multiplication(text)

    # Greek letters and other common identifiers
    greek = {
        "alpha": "alpha",
        "beta": "beta",
        "gamma": "gamma",
        "delta": "delta",
        "epsilon": "epsilon",
        "varepsilon": "varepsilon",
        "zeta": "zeta",
        "eta": "eta",
        "theta": "theta",
        "vartheta": "vartheta",
        "iota": "iota",
        "kappa": "kappa",
        "lambda": "lam_",
        "mu": "mu",
        "nu": "nu",
        "xi": "xi",
        "pi": "pi",
        "varpi": "varpi",
        "rho": "rho",
        "varrho": "varrho",
        "sigma": "sigma",
        "varsigma": "varsigma",
        "tau": "tau",
        "upsilon": "upsilon",
        "phi": "phi",
        "varphi": "varphi",
        "chi": "chi",
        "psi": "psi",
        "omega": "omega",
    }
    for g, sympy_name in greek.items():
        text = re.sub(r"\\" + g + r"\b", sympy_name, text)

    return text


def _add_implicit_multiplication(text: str) -> str:
    """Insert ``*`` between things that sympy's ``parse_expr`` doesn't auto-join."""
    # Number followed by letter or `(` or Greek letter
    text = re.sub(r"(\d)([A-Za-z_(])", r"\1*\2", text)
    # `)` followed by `(` or letter or digit
    text = re.sub(r"(\))(\()", r"\1*\2", text)
    text = re.sub(r"(\))([A-Za-z0-9_])", r"\1*\2", text)
    # Letter followed by `(`, but only if the letter is a single identifier
    # character.  We do this very conservatively to avoid mangling things like
    # `sin(`.
    text = re.sub(r"(\b[A-Za-z_][A-Za-z0-9_]*)\s+(?=[A-Za-z_(])", r"\1*", text)
    return text


def _expand_frac(text: str) -> str:
    """Replace ``\\frac{a}{b}`` (and ``\\dfrac``/``\\tfrac``) with ``(a)/(b)``.

    Special case: ``\\frac{d}{dvar}`` is interpreted as the derivative
    ``Derivative(body, var)`` once the body is read.  We don't do the
    body substitution here because the body lives outside the frac.
    """
    out = []
    i = 0
    while i < len(text):
        m = re.search(r"\\(frac|dfrac|tfrac)\s*\{", text[i:])
        if not m:
            out.append(text[i:])
            return "".join(out)
        start = i + m.start()
        out.append(text[i:start])
        # Open brace for numerator
        j = i + m.end() - 1
        num, j_after_num = _read_brace_block(text, j)
        if num is None:
            out.append(text[start:i + m.end()])
            i = i + m.end()
            continue
        # Skip whitespace
        while j_after_num < len(text) and text[j_after_num].isspace():
            j_after_num += 1
        if j_after_num >= len(text) or text[j_after_num] != "{":
            out.append(text[start:j_after_num])
            i = j_after_num
            continue
        den, j_after_den = _read_brace_block(text, j_after_num)
        if den is None:
            out.append(text[start:j_after_num])
            i = j_after_num
            continue
        # Detect ``\frac{d}{dvar}`` and stash the variable in den for the
        # body reader to pick up later via a sentinel.
        den_stripped = den.strip()
        m_dvar = re.match(r"^d\s*([A-Za-z])\s*$", den_stripped)
        if num.strip() == "d" and m_dvar:
            # Emit a sentinel that ``_expand_derivative`` replaces after
            # the body has been read (look for ``__DERIV_<var>__``).
            out.append(f"__DERIV_{m_dvar.group(1)}__")
            i = j_after_den
            continue
        out.append(f"({num})/({den})")
        i = j_after_den
    return "".join(out)


def _expand_derivative_sentinel(text: str) -> str:
    """Find the body after a ``__DERIV_<var>__`` sentinel and emit ``Derivative(body, var)``.

    The body is read up to end-of-string or a top-level ``+``/``-`` /
    ``=`` operator (so terms like ``\\frac{d}{dx} x^2 + y^2`` become
    ``Derivative(x**2, x) + y**2``).
    """
    sentinel_re = re.compile(r"__DERIV_([A-Za-z])__")
    out = []
    i = 0
    while i < len(text):
        m = sentinel_re.search(text, i)
        if not m:
            out.append(text[i:])
            break
        start = m.start()
        var = m.group(1)
        out.append(text[i:start])
        j = m.end()
        body, j_after = _read_derivative_body(text, j)
        if body is None:
            out.append(text[start:j])
            i = j
            continue
        out.append(f"Derivative({body}, {var})")
        i = j_after
    return "".join(out)


def _read_derivative_body(text: str, start: int) -> Tuple[Optional[str], int]:
    """Read the body following a ``\\frac{d}{dx}`` sentinel."""
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text):
        return None, start
    body, j = _read_balanced_expression(text, start)
    return body, j


def _expand_sum(text: str) -> str:
    """Replace ``\\sum_{v=lo}^{hi} BODY`` with ``Sum(BODY, (v, lo, hi))``.

    Tolerant of:
    * ``\\sum_{v}^{hi} BODY``  (no ``=lo``; defaults to 0)
    * ``\\sum_{v=lo} BODY``     (no upper bound; defaults to ``oo``)
    * ``\\sum BODY``              (no bounds — we still wrap if ``BODY`` is small)
    * ``\\sum{v}^{hi}`` (no leading underscore — LaTeX accepts this)
    * BODY in ``{...}``, ``(...)``, an identifier, a function call, or a
      full expression terminated by ``\\``, end-of-string, a closing
      brace, or an operator that clearly ends the sum (e.g. ``+``, ``-``,
      ``\\cdot``, ``\\times``, ``=``).
    """
    out = []
    i = 0
    # The leading ``_`` is optional because some authors write
    # ``\\sum{...}`` instead of ``\\sum_{...}``.  We deliberately omit
    # ``\\b`` because ``_`` (used in identifiers like ``\\sum_{i=0}``) is
    # itself a word character, which would prevent the boundary from
    # matching at the underscore.
    head_re = re.compile(
        r"\\sum"
        r"(?:\s*_?\s*(?:\{([^{}]+)\}|(\\[A-Za-z]+|[A-Za-z][A-Za-z0-9_]*)))?"
        r"(?:\s*\^\s*(?:\{([^{}]+)\}|(\\[A-Za-z]+|[A-Za-z][A-Za-z0-9_]*|\d+)))?"
    )
    while i < len(text):
        m = head_re.search(text, i)
        if not m:
            out.append(text[i:])
            break
        start = m.start()
        sub_text = m.group(1) or m.group(2) or ""
        sup_text = m.group(3) or m.group(4) or "oo"
        out.append(text[i:start])
        j = m.end()
        # Consume optional whitespace.
        while j < len(text) and text[j].isspace():
            j += 1
        # If the sum is wrapped in user parens, the body extends to the
        # matching ``)``.
        body = None
        j_after = j
        k = start - 1
        while k >= 0 and text[k].isspace():
            k -= 1
        if k >= 0 and text[k] == "(":
            depth = 1
            p = k + 1
            while p < len(text) and depth > 0:
                if text[p] == "(":
                    depth += 1
                elif text[p] == ")":
                    depth -= 1
                p += 1
            if depth == 0:
                paren_close = p - 1
                if paren_close > j:
                    body = text[j:paren_close].strip()
                    j_after = paren_close
        if body is None:
            body, j_after = _read_sum_body(text, j)
            if body is None:
                # No explicit body — e.g. ``\sum_a`` (degenerate sum where
                # the bound variable is also the summand).  Fall back to the
                # bound variable as the body so the LaTeX still parses.
                var, lower = _split_var_lower(sub_text)
                if var:
                    out.append(f"Sum({var}, ({var}, {lower}, {sup_text.strip() or 'oo'}))")
                    i = j
                    continue
                out.append(text[start:j])
                i = j
                continue
        var, lower = _split_var_lower(sub_text)
        upper = sup_text.strip() or "oo"
        out.append(f"Sum({body}, ({var}, {lower}, {upper}))")
        i = j_after
    return "".join(out)


def _expand_int(text: str) -> str:
    """Replace ``\\int_{lo}^{hi} BODY\\,dvar`` with ``Integral(BODY, (var, lo, hi))``.

    Tolerates missing ``dvar`` at the end (we treat the last identifier
    in the body as the integration variable) and missing bounds
    (defaults to ``(-oo, oo)``).
    """
    head_re = re.compile(
        r"\\int"
        r"(?:\s*_\s*(?:\{([^{}]+)\}|(\\[A-Za-z]+|[A-Za-z][A-Za-z0-9_]*|\d+)))?"
        r"(?:\s*\^\s*(?:\{([^{}]+)\}|(\\[A-Za-z]+|[A-Za-z][A-Za-z0-9_]*|\d+)))?"
    )
    out = []
    i = 0
    while i < len(text):
        m = head_re.search(text, i)
        if not m:
            out.append(text[i:])
            break
        start = m.start()
        sub_text = m.group(1) or m.group(2) or ""
        sup_text = m.group(3) or m.group(4) or "oo"
        out.append(text[i:start])
        j = m.end()
        # Consume optional whitespace.
        while j < len(text) and text[j].isspace():
            j += 1
        # If the integral is wrapped in user parens — i.e. the character
        # immediately before ``\int`` (skipping spaces) is ``(`` — the body
        # extends to the matching ``)``.  This lets users write
        # ``(\int_a^b 2x + b)`` to mean "integrate the whole ``2x + b``".
        body = None
        j_after = j
        k = start - 1
        while k >= 0 and text[k].isspace():
            k -= 1
        if k >= 0 and text[k] == "(":
            depth = 1
            p = k + 1
            while p < len(text) and depth > 0:
                if text[p] == "(":
                    depth += 1
                elif text[p] == ")":
                    depth -= 1
                p += 1
            if depth == 0:
                paren_close = p - 1
                if paren_close > j:
                    body = text[j:paren_close].strip()
                    j_after = paren_close
        if body is None:
            body, j_after = _read_sum_body(text, j)
            if body is None:
                out.append(text[start:j])
                i = j
                continue
        # The integration variable is the bit after ``d`` if present, else
        # we fall back to the subscript variable (single-letter case).
        var, lower, upper = _split_int_bounds(sub_text, sup_text)
        dvar_match = re.search(r"\s*\\?d\s*([A-Za-z])\b", body)
        if dvar_match:
            var = dvar_match.group(1)
            # Strip the ``dx`` from the body so the integrand is what the
            # user actually wrote (without the differential).
            body = body[: dvar_match.start()] + body[dvar_match.end():]
        out.append(f"Integral({body}, ({var}, {lower}, {upper}))")
        i = j_after
    return "".join(out)


def _split_int_bounds(sub_text: str, sup_text: str) -> Tuple[str, str, str]:
    """Pull ``var, lower, upper`` from ``\\int``'s subscript/superscript.

    Defaults: ``var='x'`` (later replaced from the ``dvar`` in the body),
    ``lower='-oo'``, ``upper='oo'``.

    Forms accepted:
      * ``\\int_{x=a}^{b}``  -> var='x', lower='a', upper='b'
      * ``\\int_{a}^{b}``    -> lower='a', upper='b' (var from differential)
      * ``\\int_a^b``        -> same as above, no braces
      * ``\\int``            -> bounds default to (-oo, oo)
    """
    upper = sup_text.strip() or "oo"
    sub_text = sub_text.strip()
    if not sub_text:
        return "x", "-oo", upper or "oo"
    if "=" in sub_text:
        var, lower = sub_text.split("=", 1)
        return var.strip() or "x", lower.strip() or "-oo", upper or "oo"
    # No `=` in subscript: treat the whole subscript as the lower bound
    # (the variable will come from the ``dvar`` in the body).
    return "x", sub_text or "-oo", upper or "oo"


def _expand_lim(text: str) -> str:
    """Replace ``\\lim_{var \\to value} BODY`` with ``Limit(BODY, var, value)``.

    Accepts ``\\to``, ``,`` or ``=`` as the variable / value separator.
    """
    head_re = re.compile(
        r"\\lim"
        r"(?:\s*_?\s*\{([^{}]+)\})?"
    )
    out = []
    i = 0
    while i < len(text):
        m = head_re.search(text, i)
        if not m:
            out.append(text[i:])
            break
        start = m.start()
        # ``\to`` and friends live inside the subscript — split here so we
        # don't depend on the macro replacements running first.
        sub_text = (m.group(1) or "").strip()
        sub_text = re.sub(r"\\to\b|\\rightarrow\b|\\Rightarrow\b|=", ",", sub_text)
        out.append(text[i:start])
        j = m.end()
        while j < len(text) and text[j].isspace():
            j += 1
        body, j_after = _read_sum_body(text, j)
        if body is None:
            out.append(text[start:j])
            i = j
            continue
        parts = [p.strip() for p in sub_text.split(",") if p.strip()]
        if len(parts) >= 2:
            var, value = parts[0], parts[1]
        else:
            var, value = (parts[0] if parts else "x"), "0"
        out.append(f"Limit({body}, {var}, {value})")
        i = j_after
    return "".join(out)


def _split_var_lower(sub_text: str) -> Tuple[str, str]:
    """Parse ``var=lo`` style subscript into ``(var, lo)``; default lo=0."""
    sub_text = sub_text.strip()
    if "=" in sub_text:
        var, lower = sub_text.split("=", 1)
        return var.strip(), lower.strip()
    return sub_text, "0"


def _read_sum_body(text: str, start: int) -> Tuple[Optional[str], int]:
    """Best-effort read of the body of a ``\\sum`` / ``\\prod`` command.

    Returns ``(body, position_after)`` or ``(None, start)`` if it cannot
    find anything sensible.
    """
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text):
        return None, start
    # 1) Brace block: { ... }
    if text[start] == "{":
        body, j = _read_brace_block(text, start)
        if body is not None:
            # Keep the outer braces so Python doesn't see them as a set
            # literal; downstream preprocessing strips them safely.  Also
            # keep any trailing token (``^expo``, `` dx``) so we don't drop
            # the differential after a braced integrand.
            rest, rest_end = _read_balanced_expression(text, j)
            if rest_end > j and rest is not None:
                return text[start:rest_end].strip(), rest_end
            return text[start:j], j
    # 2) Paren block: ( ... )
    if text[start] == "(":
        body, j = _read_paren_block(text, start)
        if body is not None:
            # Keep any trailing token (``^expo``, `` dx``) so we don't drop
            # the differential after a parenthesised integrand.
            rest, rest_end = _read_balanced_expression(text, j)
            if rest_end > j and rest is not None:
                return text[start:rest_end].strip(), rest_end
            return text[start:j], j
    # 3) A backslash macro possibly followed by parens: \\foo or \\foo(...)
    if text[start] == "\\":
        m = re.match(r"\\[A-Za-z]+", text[start:])
        if m:
            j = start + m.end()
            if j < len(text) and text[j] == "(":
                # Read the parenthesised argument list and keep the whole
                # ``\foo(args)`` form so the caller sees a complete body.
                body, j_after = _read_paren_block(text, j)
                if body is not None:
                    # If there's more text after the parens (e.g. a
                    # differential ``\sin(x) dx``), keep reading so we
                    # don't drop the trailing bits.
                    rest, rest_end = _read_balanced_expression(text, j_after)
                    if rest_end > j_after and rest is not None:
                        return text[start:rest_end].strip(), rest_end
                    return text[start:j_after], j_after
            # If the macro is itself a ``\sum`` / ``\prod`` (nested),
            # capture it together with any subscript/superscript and let
            # the caller re-enter ``_expand_sum`` so the entire nested
            # expression gets wrapped.
            inner = text[start:j]
            rest = text[j:]
            if inner in ("\\sum", "\\prod") and (
                rest.startswith("_{") or rest.startswith("^")
            ):
                body, j_after = _read_balanced_expression(text, start)
                if body is not None and body.lstrip().startswith("\\sum"):
                    body = _expand_sum(body)
                elif body is not None and body.lstrip().startswith("\\prod"):
                    body = _expand_prod(body)
                return body, j_after
            # If there's more text after the macro (e.g. ``\sin x dx``),
            # include it in the body so callers can detect the differential.
            rest, rest_end = _read_balanced_expression(text, j)
            if rest_end > j and rest is not None:
                return text[start:rest_end].strip(), rest_end
            return inner, j
    # 4) Single identifier or function call (optionally followed by ``^`` and
    #    a sub-expression so things like ``x^2 dx`` are read as one body).
    m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[start:])
    if m:
        j = start + m.end()
        if j < len(text) and text[j] == "(":
            body, j_after = _read_paren_block(text, j)
            if body is not None:
                return body, j_after
            return text[start:j], j
        if j < len(text) and text[j] == "^":
            # Consume the exponent so the body extends to ``x^2`` rather than
            # just ``x`` (relevant for ``\int ... x^2 dx``).
            j += 1
            if j < len(text) and text[j] == "{":
                _, j = _read_brace_block(text, j)
                if j < 0:
                    j = start + m.end()
            else:
                _, j = _read_balanced_expression(text, j)
            # After the exponent, only continue reading if the next
            # non-whitespace char is NOT a top-level + or - (which would
            # mean the integrand/summand ended).  This keeps ``x^2 dx``
            # together but stops at ``x^2 + b``.
            k = j
            while k < len(text) and text[k].isspace():
                k += 1
            if k < len(text) and text[k] in "+-":
                return text[start:j].strip(), j
            body, body_end = _read_balanced_expression(text, j)
            if body_end > j and body is not None:
                return (text[start:body_end].strip(), body_end)
            return text[start:j].strip(), j
        # Identifier followed by whitespace + more text — fall through to
        # the balanced-expression reader so multi-token bodies like
        # ``x dx`` are read as one.
    # 5) Multi-token expression: read until end-of-string or the next
    #    *closing brace / paren / bracket* at depth 0 or an operator that
    #    clearly ends the sum (top-level + or -, top-level =).
    body, j = _read_balanced_expression(text, start)
    # If the body starts with another ``\sum`` / ``\prod``, recurse so the
    # nested sum gets expanded too (e.g. ``\sum_i \sum_j a_i b_j``).
    if body is not None and body.lstrip().startswith("\\sum"):
        body = _expand_sum(body)
    elif body is not None and body.lstrip().startswith("\\prod"):
        body = _expand_prod(body)
    return body, j


def _read_balanced_expression(text: str, start: int) -> Tuple[Optional[str], int]:
    """Read a math expression with balanced parens/braces/brackets.

    Stops at end-of-string or at a top-level ``+``, ``-`` or ``=`` that
    clearly delimits the expression.  ``*``, ``/`` and ``\\cdot`` do
    NOT terminate since they bind tighter than ``+``.
    """
    depth = 0
    j = start
    saw_anything = False
    while j < len(text):
        ch = text[j]
        if ch in "({[":
            depth += 1
            saw_anything = True
            j += 1
            continue
        if ch in ")}]":
            if depth == 0:
                break
            depth -= 1
            j += 1
            saw_anything = True
            continue
        # Top-level operator that ends the expression.
        if depth == 0 and ch in "+-":
            # Only stop if we've already consumed something AND the next
            # character is not the same operator (which would mean ``x--y``
            # or ``a+-b`` style negation).
            if saw_anything:
                nxt = text[j + 1] if j + 1 < len(text) else ""
                if nxt != ch:
                    break
        if depth == 0 and ch == "=":
            # Top-level ``=`` means the sum's body ended and we're back
            # at the formula's LHS/RHS separator; bail.
            if saw_anything:
                break
        j += 1
        saw_anything = True
    if not saw_anything:
        return None, start
    return text[start:j].strip(), j


def _read_paren_block(text: str, start: int) -> Tuple[Optional[str], int]:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "(":
        return None, start
    depth = 0
    for k in range(start, len(text)):
        ch = text[k]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:k], k + 1
    return None, start


def _expand_prod(text: str) -> str:
    """Mirror of :func:`_expand_sum` for ``\\prod``."""
    out = []
    i = 0
    head_re = re.compile(
        r"\\prod"
        r"(?:\s*_?\s*(?:\{([^{}]+)\}|(\\[A-Za-z]+|[A-Za-z][A-Za-z0-9_]*)))?"
        r"(?:\s*\^\s*(?:\{([^{}]+)\}|(\\[A-Za-z]+|[A-Za-z][A-Za-z0-9_]*|\d+)))?"
    )
    while i < len(text):
        m = head_re.search(text, i)
        if not m:
            out.append(text[i:])
            break
        start = m.start()
        sub_text = m.group(1) or m.group(2) or ""
        sup_text = m.group(3) or m.group(4) or "oo"
        out.append(text[i:start])
        j = m.end()
        while j < len(text) and text[j].isspace():
            j += 1
        body = None
        j_after = j
        k = start - 1
        while k >= 0 and text[k].isspace():
            k -= 1
        if k >= 0 and text[k] == "(":
            depth = 1
            p = k + 1
            while p < len(text) and depth > 0:
                if text[p] == "(":
                    depth += 1
                elif text[p] == ")":
                    depth -= 1
                p += 1
            if depth == 0:
                paren_close = p - 1
                if paren_close > j:
                    body = text[j:paren_close].strip()
                    j_after = paren_close
        if body is None:
            body, j_after = _read_sum_body(text, j)
            if body is None:
                out.append(text[start:j])
                i = j
                continue
        var, lower = _split_var_lower(sub_text)
        upper = sup_text.strip() or "oo"
        out.append(f"Product({body}, ({var}, {lower}, {upper}))")
        i = j_after
    return "".join(out)


def _expand_sqrt(text: str) -> str:
    out = []
    i = 0
    while True:
        m = re.search(r"\\sqrt\s*\{", text[i:])
        if not m:
            out.append(text[i:])
            return "".join(out)
        start = i + m.start()
        out.append(text[i:start])
        body, j_after = _read_brace_block(text, i + m.end() - 1)
        if body is None:
            out.append(text[start:i + m.end()])
            i = i + m.end()
            continue
        out.append(f"sqrt({body})")
        i = j_after


def _expand_abs(text: str) -> str:
    """Replace ``|expr|`` (absolute value bars) with ``Abs(expr)``.

    Walks the string tracking brace/paren depth so that vertical bars
    that appear inside other delimiters (or inside ``||`` already) are
    not misinterpreted as absolute-value markers.  ``\\|`` is treated
    as a literal ``|`` (LaTeX norm delimiter) so the inner expression
    doesn't include the stray backslash.
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n and text[i + 1] == "|":
            # ``\|`` (LaTeX norm) — just keep as ``|`` so the next
            # iteration sees the closing ``|``.
            out.append("|")
            i += 2
            continue
        if ch == "|":
            # Skip if the previous output char was also ``|`` (``||``).
            if out and out[-1] == "|":
                # Already emitted as part of a previous ``\|`` — skip.
                i += 1
                continue
            # Find the matching closing ``|`` at the same depth.
            j = i + 1
            depth = 0
            while j < n:
                cj = text[j]
                if cj == "\\" and j + 1 < n and text[j + 1] == "|":
                    # ``\|`` — treat as a single ``|`` token.
                    j += 2
                    continue
                if cj in "({[":
                    depth += 1
                elif cj in ")}]":
                    depth -= 1
                elif cj == "|" and depth == 0:
                    break
                j += 1
            if j >= n:
                # Unmatched — just keep the original.
                out.append(text[i:])
                return "".join(out)
            inner = text[i + 1:j].strip()
            if not inner:
                out.append("|")
            else:
                out.append(f"Abs({inner})")
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _read_brace_block(text: str, start: int) -> Tuple[Optional[str], int]:
    """Read a balanced ``{...}`` block starting at ``start``.

    Returns ``(content, position_after_closing_brace)`` or ``(None, start)``
    if no balanced block can be found.
    """
    # Skip whitespace
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        return None, start
    depth = 0
    for k in range(start, len(text)):
        ch = text[k]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:k], k + 1
    return None, start


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _default_local_dict() -> Dict[str, "Expr"]:
    """Provide a ``local_dict`` that overrides sympy's built-ins.

    sympy ships several names that clash with common ML / physics
    variables — most importantly ``gamma`` (the Gamma function), which
    prevents formulas like ``gamma * q_next`` from parsing.  By binding
    these names to plain ``Symbol`` placeholders, the parser accepts
    the user's free variables.
    """
    from sympy import Symbol
    return {
        "gamma": Symbol("gamma"),
        "Gamma": Symbol("Gamma"),
        "Beta": Symbol("Beta"),
        "Zeta": Symbol("Zeta"),
        "E": Symbol("E"),
        "N": Symbol("N"),
    }


def parse_infix(expr_str: str, local_dict: Optional[Dict[str, "Expr"]] = None) -> "Expr":
    """Parse an infix expression via ``sympy.parse_expr``.

    Users often paste LaTeX-like syntax into the infix box (``|x - y|``,
    ``\\sigma(z)``, etc.), so we apply the same abs / function-name
    preprocessors as the LaTeX path before handing the result to
    ``parse_expr``.
    """
    transformations = standard_transformations + (implicit_multiplication_application,)
    merged = _default_local_dict()
    if local_dict:
        merged.update(local_dict)
    text = expr_str
    text = _expand_abs(text)
    # Strip ``\func`` so ``\sigma(z)`` parses like ``sigma(z)``.
    text = re.sub(
        r"\\(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|exp|log|ln|sqrt|abs|sigma|sign|min|max|erf|sigmoid|softmax|relu|leakyrelu|tanh|hardtanh|softplus|gelu|mish|step)\b",
        r"\1",
        text,
    )
    return parse_expr(
        text,
        local_dict=merged,
        transformations=transformations,
        evaluate=True,
    )


# Common LHS prefixes like ``f(x) =``, ``f(x, y) =``, ``f =``, ``y =`` that users
# often type.  We also handle LaTeX macro-wrapped names like
# ``\\mathrm{H}(p, q) = ...`` (macro + braced argument + argument list) or
# ``\\sigma(z) = ...`` (bare macro + argument list).
# Note: we explicitly allow commas inside the parentheses so things like
# ``f(x, y) = ...`` strip correctly.
_LHS_PREFIX_RE = re.compile(
    # 1. ``\\macro{X}(args) = ...`` (macro with braced argument).
    r"^\s*\\[A-Za-z]+\s*\{[A-Za-z][A-Za-z0-9_]*\}\s*\([^)]*\)\s*=\s*"
    # 2. ``\\macro(args) = ...`` (bare macro + argument list).
    r"|^\s*\\[A-Za-z]+\s*\([^)]*\)\s*=\s*"
    # 3. ``identifier(args) = ...`` (plain identifier).
    r"|^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*=\s*"
    # 4. ``\\macro{X} = ...`` (macro with braced argument, no arg list).
    r"|^\s*\\[A-Za-z]+\s*\{[A-Za-z][A-Za-z0-9_]*\}\s*=\s*"
    # 5. ``\\macro = ...`` (bare macro).
    r"|^\s*\\[A-Za-z]+\s*=\s*"
    # 6. ``identifier = ...`` (plain identifier).
    r"|^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*"
)


def strip_lhs_assignment(text: str) -> str:
    """Strip a leading ``f(x) =`` / ``y =`` / ``f =`` style assignment.

    Users frequently paste formulas like ``f(x) = sin(a*x) + b`` or
    ``y = a*x + b``.  Both ``parse_expr`` and ``parse_latex`` choke on the
    ``=`` because Python/sympy treat it as invalid syntax.  Stripping the
    left-hand side is a tiny, predictable transformation.
    """
    return _LHS_PREFIX_RE.sub("", text, count=1).strip()


def parse_latex_or_infix(text: str, *, mode: str = "auto") -> "Expr":
    """Parse either LaTeX or infix (auto-detect unless ``mode`` overrides).

    Parameters
    ----------
    text:
        The formula text.
    mode:
        ``"auto"`` (default), ``"latex"`` or ``"infix"``.
    """
    if not text or not text.strip():
        raise ValueError("Empty formula")

    text = strip_lhs_assignment(text)

    if mode == "auto":
        mode = "latex" if "\\" in text else "infix"

    if mode == "infix":
        return parse_infix(text)

    # Try the proper LaTeX parser first (it knows about every macro).
    try:
        from sympy.parsing.latex import parse_latex as _parse_latex  # type: ignore
    except Exception:
        _parse_latex = None  # type: ignore

    if _parse_latex is not None:
        try:
            return _parse_latex(text)
        except Exception:
            # Fall through to our manual preprocessor + parse_expr.
            pass

    preprocessed = preprocess_latex(text)
    return parse_infix(preprocessed)


# ---------------------------------------------------------------------------
# Hyperparameter extraction
# ---------------------------------------------------------------------------

@dataclass
class SuggestedParameter:
    """A hyperparameter suggested from a parsed formula."""

    name: str
    kind: str = "range"  # one of: range, fixed, choice
    lower: Optional[float] = None
    upper: Optional[float] = None
    value_type: str = "float"  # int or float
    log_scale: bool = False
    note: str = ""

    def to_omniopt_arg(self) -> str:
        """Serialise as ``--parameter`` argument (string form)."""
        if self.kind == "fixed":
            return f"{self.name} fixed {self.value}"
        if self.kind == "range":
            lower = "0" if self.lower is None else _fmt(self.lower)
            upper = "1" if self.upper is None else _fmt(self.upper)
            log = "true" if self.log_scale else "false"
            return f"{self.name} range {lower} {upper} {self.value_type} {log}"
        # choice
        return f"{self.name} choice {self.value}"

    # convenience alias for "fixed" form
    @property
    def value(self) -> str:
        return "0" if self.lower is None else _fmt(self.lower)


def _fmt(num: float) -> str:
    if isinstance(num, Integer):
        return str(int(num))
    if isinstance(num, Float) and num == int(num):
        return str(int(num))
    return repr(float(num))


# Names we should never treat as hyperparameters (mathematical constants and
# sympy built-ins).
_DEFAULT_RESERVED = frozenset(
    {
        # Greek letters and other sympy constants
        "oo", "inf", "infty", "nan", "NaN",
        "zoo", "N", "Z", "Q", "R", "C", "S", "True", "False",
        "I",  # sympy's imaginary unit
        # common math funcs
        "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh",
        "exp", "log", "ln", "sqrt", "abs",
        "Min", "Max",
    }
)

# Common mathematical constants that we want to suggest as **fixed**
# parameters with sensible default values.  The user can always change
# ``kind`` to ``range`` in the GUI if they want to optimise them.
_KNOWN_CONSTANTS = {
    "e": (2.718281828459045, "Euler's number"),
    "E": (2.718281828459045, "Euler's number (sympy convention)"),
    "pi": (3.141592653589793, "pi"),
    "PI": (3.141592653589793, "pi (uppercase)"),
}


def suggest_hyperparameters(
    expr: "Expr",
    reserved: Optional[Iterable[str]] = None,
    default_range_low: float = -1.0,
    default_range_high: float = 1.0,
) -> List[SuggestedParameter]:
    """Return a list of suggested :class:`SuggestedParameter` for the free symbols of ``expr``.

    We deliberately only produce *suggestions*.  The user is expected to
    edit them in the GUI before launching the run.

    The heuristic:

    * Take every free symbol of the expression.
    * Drop anything that is in ``reserved`` (defaulting to sympy functions
      and infinity/zoo/etc.).
    * If the symbol is a known mathematical constant (``e``, ``pi``), suggest
      it as a **fixed** parameter with its conventional default value.
    * For names ending in ``_int`` we suggest ``int``, otherwise ``float``.
    * We default to a ``[-1, 1]`` range with no log scale.  Names that start
      with ``lr_``, ``log_`` or end with ``_log`` get ``log_scale=True`` and
      a positive default range ``[1e-5, 1e-1]``.
    """
    if reserved is None:
        reserved = set()
    reserved_set = set(_DEFAULT_RESERVED) | set(reserved)

    free_symbols = sorted(expr.free_symbols, key=lambda s: s.name)

    suggestions: List[SuggestedParameter] = []
    for sym in free_symbols:
        name = sym.name
        if name in reserved_set:
            continue

        if name in _KNOWN_CONSTANTS:
            value, _note = _KNOWN_CONSTANTS[name]
            suggestions.append(
                SuggestedParameter(
                    name=name,
                    kind="fixed",
                    lower=value,
                    upper=value,
                    value_type="float",
                    log_scale=False,
                    note=f"fixed constant ({_note}); user can switch to range",
                )
            )
            continue

        if name.endswith("_int"):
            value_type = "int"
            lower = 0.0
            upper = 10.0
        elif name.startswith("lr_") or name.startswith("log_") or name.endswith("_log"):
            value_type = "float"
            lower = 1e-5
            upper = 1e-1
            log_scale = True
        else:
            value_type = "float"
            lower = default_range_low
            upper = default_range_high
            log_scale = False

        suggestions.append(
            SuggestedParameter(
                name=name,
                kind="range",
                lower=lower,
                upper=upper,
                value_type=value_type,
                log_scale=log_scale,
                note="auto-suggested from formula",
            )
        )
    return suggestions


def suggest_formula_split(
    raw_text: str,
    expr: "Expr",
    *,
    reserved: Optional[Iterable[str]] = None,
    default_range_low: float = -1.0,
    default_range_high: float = 1.0,
) -> "Tuple[List[SuggestedParameter], List[SuggestedParameter], List[str]]":
    """Split ``raw_text`` into parameters and constants based on the LHS.

    Returns ``(parameters, constants, bound_names)``.

    The heuristic mirrors the JS-side client-side parser:

    * Find the LHS of an ``=`` assignment.  Inside the outermost
      parentheses (if any), the identifiers become *parameters* (range).
    * In the RHS, identifiers that appear in the LHS remain parameters;
      identifiers that only appear in the RHS become *constants* (fixed)
      unless they are bound by ``\\sum_`` / ``\\prod_``.
    * Well-known constants (``e``, ``pi``) are always suggested as fixed
      parameters with their conventional value.
    * If no LHS exists, every free symbol is treated as a parameter.
    """
    if reserved is None:
        reserved = set()
    reserved_set = set(_DEFAULT_RESERVED) | set(reserved)

    lhs, _ = _split_assignment(raw_text)
    bound_names = set(_find_bound_names(raw_text))

    # Parameter names: identifiers inside the outermost parentheses of the
    # LHS (or every identifier in the LHS if there are no parens).
    lhs_param_names: List[str] = []
    if lhs.strip():
        # Use sympy to parse the parameter list (e.g. "x" or "x, y" or even
        # nested calls — we take the outermost).
        try:
            param_str = _outermost_paren_content(lhs)
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", param_str):
                if tok in reserved_set:
                    continue
                if tok not in lhs_param_names:
                    lhs_param_names.append(tok)
        except Exception:
            pass

    # Now collect identifiers that actually appear in the RHS expression.
    # We use the parsed expression (which already excludes the LHS) for
    # correctness; the raw_text parameter names only affect whether an
    # RHS symbol is promoted to "parameter" or kept as "constant".
    rhs_free = sorted({s.name for s in expr.free_symbols} - reserved_set)

    parameters: List[SuggestedParameter] = []
    constants: List[SuggestedParameter] = []

    def _as_range_suggestion(name: str) -> SuggestedParameter:
        if name.endswith("_int"):
            return SuggestedParameter(name=name, kind="range", lower=0.0, upper=10.0, value_type="int", log_scale=False)
        if name.startswith("lr_") or name.startswith("log_") or name.endswith("_log"):
            return SuggestedParameter(name=name, kind="range", lower=1e-5, upper=1e-1, value_type="float", log_scale=True)
        return SuggestedParameter(name=name, kind="range", lower=default_range_low, upper=default_range_high, value_type="float", log_scale=False)

    def _as_constant_suggestion(name: str) -> SuggestedParameter:
        if name in _KNOWN_CONSTANTS:
            value, _note = _KNOWN_CONSTANTS[name]
            return SuggestedParameter(
                name=name,
                kind="fixed",
                lower=value,
                upper=value,
                value_type="float",
                log_scale=False,
                note=f"fixed constant ({_note}); switch to range to optimize",
            )
        return SuggestedParameter(
            name=name,
            kind="fixed",
            lower=1.0,
            upper=1.0,
            value_type="float",
            log_scale=False,
            note="constant (appears only in RHS); user should set the value",
        )

    lhs_set = set(lhs_param_names)
    if lhs_param_names:
        for n in lhs_param_names:
            if n in bound_names:
                continue
            if n in reserved_set:
                continue
            if n in _KNOWN_CONSTANTS:
                parameters.append(_as_constant_suggestion(n))
            else:
                parameters.append(_as_range_suggestion(n))
        for n in rhs_free:
            if n in lhs_set:
                continue
            if n in bound_names:
                continue
            if n in reserved_set:
                continue
            constants.append(_as_constant_suggestion(n))
    else:
        # No LHS — everything is a parameter.
        for n in rhs_free:
            if n in bound_names:
                continue
            if n in reserved_set:
                continue
            if n in _KNOWN_CONSTANTS:
                parameters.append(_as_constant_suggestion(n))
            else:
                parameters.append(_as_range_suggestion(n))

    return parameters, constants, sorted(bound_names)


def _split_assignment(text: str) -> "Tuple[str, str]":
    """Return ``(lhs, rhs)`` from ``text`` split on the first top-level ``=``.

    Returns ``("", text)`` if no top-level ``=`` is present.
    """
    depth = 0
    for i, ch in enumerate(text):
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        elif ch == "=" and depth == 0:
            return text[:i], text[i + 1:]
    return "", text


def _outermost_paren_content(text: str) -> str:
    """Return the content of the *last* top-level ``(...)`` group in ``text``."""
    open_idx = text.rfind("(")
    if open_idx < 0:
        return text
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
    return text[open_idx + 1:]


def _find_bound_names(text: str) -> "List[str]":
    """Find identifier names bound by ``\\sum_{...}`` / ``\\prod_{...}`` /
    ``\\int_{...}^{...}``.

    Also handles the sympy Python-style ``Sum(expr, (var, lo, hi))``,
    ``Product(expr, (var, lo, hi))`` and ``Integral(expr, (var, lo, hi))``
    calls where ``var`` is the bound name.
    """
    bound: List[str] = []
    # LaTeX form: \sum_{var=...}^{...} ..., \prod_{...}, \int_{...}^{...} ...
    for m in re.finditer(
        r"\\(?:sum|prod|int)\s*_\s*(?:\{([^{}]+)\}|([A-Za-z][A-Za-z0-9_]*))",
        text,
    ):
        inside = (m.group(1) or m.group(2) or "").strip()
        if "=" in inside:
            inside = inside.split("=", 1)[0].strip()
        if "," in inside:
            inside = inside.split(",", 1)[0].strip()
        if inside and inside not in bound:
            bound.append(inside)
    # Python sympy form: Sum/Product/Integral(expr, (var, lo, hi), ...)
    for m in re.finditer(
        r"\b(?:Sum|Product|Integral)\s*\(",
        text,
    ):
        start = m.end()
        depth = 1
        i = start
        args: List[str] = []
        buf: List[str] = []
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "(":
                depth += 1
                buf.append(ch)
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    args.append("".join(buf).strip())
                    break
                buf.append(ch)
            elif ch == "," and depth == 1:
                args.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
            i += 1
        # The second argument is (var, lo, hi).
        if len(args) >= 2 and args[1].startswith("(") and args[1].endswith(")"):
            inner = args[1][1:-1]
            parts = [p.strip() for p in inner.split(",")]
            if parts:
                name = parts[0]
                if name and name not in bound:
                    bound.append(name)
    return bound


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_ascii(expr: "Expr") -> str:
    """Render ``expr`` as ASCII via :func:`sympy.pretty`."""
    return sympy_pretty(expr, use_unicode=False)


def render_latex(expr: "Expr") -> str:
    """Render ``expr`` as LaTeX.

    We use ``mul_symbol=''`` so that ``2*x`` renders as ``2x`` rather
    than ``2 x`` — much easier to read inline in the share viewer.
    """
    return sympy_latex(expr, mul_symbol="")


def render_latex_with_underbraces(
    expr: "Expr",
    parameter_names: Sequence[str],
) -> str:
    """Render ``expr`` as LaTeX with ``\\underbrace`` for each parameter occurrence.

    The first occurrence of every symbol in ``parameter_names`` is wrapped in
    ``\\underbrace{...}_{\\text{<name>}}`` so it is clear in the GUI which
    term belongs to which hyperparameter.
    """
    base = sympy_latex(expr)
    wrapped = base
    seen: Dict[str, bool] = {name: False for name in parameter_names}

    for name in parameter_names:
        if seen.get(name):
            continue
        # Match the bare symbol ``x`` but not as a substring of another
        # identifier (e.g. ``x_int`` should not match ``x``).
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
        if pattern.search(wrapped):
            wrapped = pattern.sub(
                f"\\\\underbrace{{{name}}}_{{{{\\\\text{{{name}}}}}}}",
                wrapped,
                count=1,
            )
            seen[name] = True
    return wrapped


# ---------------------------------------------------------------------------
# Lambda building
# ---------------------------------------------------------------------------

def build_lambda(
    expr: "Expr",
    parameter_names: Sequence[str],
    *,
    modules: Optional[Sequence[str]] = None,
) -> Callable[..., float]:
    """Return a numeric call-back that takes ``parameter_names`` as kwargs.

    Uses :func:`sympy.lambdify` with ``numpy`` semantics so the call-back can
    be vectorised cheaply.  When sympy can't lambdify (e.g. free symbols that
    aren't in ``parameter_names``) we fall back to ``math`` semantics.

    Definite ``Integral`` nodes are evaluated with :meth:`sympy.Expr.doit`
    up front so the resulting lambda doesn't have to know how to compile
    sympy's symbolic integration.  Indefinite integrals (``\\int f \\,dx``
    with no upper bound) are left intact.
    """
    from sympy import Integral as _Integral

    # Walk the expression tree and pre-evaluate any ``Integral`` whose
    # integration variable has definite bounds.  Sympy can usually
    # evaluate ``\\int_{a}^{b} f(x)\\,dx`` to a closed form in the
    # other variables (``q*(b - a)`` for constant ``f``), which we want
    # the numeric call-back to compute instead of trying to compile
    # sympy's symbolic integrator.
    def _eval_integrals(e: "Expr") -> "Expr":
        if e.is_Atom:
            return e
        if isinstance(e, _Integral):
            if e.limits:
                # Definite integral (no symbolic ``oo`` or ``-oo``).  Try
                # to evaluate it; fall back to the original on failure.
                try:
                    evaluated = e.doit()
                    # If the result still contains an ``Integral`` node,
                    # sympy couldn't find a closed form — bail.
                    if not evaluated.has(_Integral):
                        return evaluated
                except Exception:
                    pass
            return e
        return e.func(*[_eval_integrals(arg) for arg in e.args])

    expr = _eval_integrals(expr)

    free = sorted(expr.free_symbols, key=lambda s: s.name)
    ordered = [s for s in free if s.name in set(parameter_names)]
    extras = [s for s in free if s.name not in set(parameter_names)]
    if extras:
        # Re-substitute extras as ``0`` so sympy doesn't try to treat them
        # as parameters.  In practice users should call this with values
        # for any ``fixed`` parameters before evaluating.
        expr = expr.subs([(s, 0) for s in extras])
        free = ordered

    if modules is None:
        modules = ["numpy"]

    fn = lambdify(ordered, expr, modules=modules)
    return _wrap_lambda(fn, ordered)


def _wrap_lambda(fn: Callable[..., float], ordered: Sequence[Symbol]) -> Callable[..., Any]:
    name_to_idx = {s.name: idx for idx, s in enumerate(ordered)}

    def wrapper(**kwargs: float) -> Any:
        args = [0.0] * len(ordered)
        for name, value in kwargs.items():
            if name in name_to_idx:
                args[name_to_idx[name]] = value
        result = fn(*args)
        # If the underlying expression returned a tuple (multi-result
        # formula in Python mode), preserve it; otherwise coerce to a
        # plain float.
        if isinstance(result, tuple):
            return tuple(float(r) for r in result)
        try:
            return float(result)
        except TypeError:
            return result

    wrapper.__doc__ = f"lambda for formula with params {[s.name for s in ordered]}"
    return wrapper


# ---------------------------------------------------------------------------
# High-level Formula wrapper
# ---------------------------------------------------------------------------

@dataclass
class Formula:
    """A parsed formula together with helpers for rendering and execution."""

    raw: str
    mode: str  # 'latex' or 'infix'
    expr: "Expr"
    suggestions: List[SuggestedParameter] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str, *, mode: str = "auto") -> "Formula":
        expr = parse_latex_or_infix(text, mode=mode)
        suggestions = suggest_hyperparameters(expr)
        actual_mode = mode
        if mode == "auto":
            actual_mode = "latex" if "\\" in text else "infix"
        return cls(raw=text, mode=actual_mode, expr=expr, suggestions=suggestions)

    def ascii(self) -> str:
        return render_ascii(self.expr)

    def latex(self) -> str:
        return render_latex(self.expr)

    def latex_with_underbraces(self, parameter_names: Optional[Sequence[str]] = None) -> str:
        names = list(parameter_names) if parameter_names is not None else [s.name for s in self.suggestions]
        return render_latex_with_underbraces(self.expr, names)

    def lambda_for(self, parameter_names: Sequence[str]) -> Callable[..., float]:
        return build_lambda(self.expr, parameter_names)


# ---------------------------------------------------------------------------
# Submitit availability helper
# ---------------------------------------------------------------------------

def choose_latex_or_infix(text: str) -> str:
    """Heuristic: return ``'latex'`` if the text contains ``\\`` else ``'infix'``."""
    return "latex" if "\\" in text else "infix"


# ---------------------------------------------------------------------------
# Module-level smoke-test (run as `python3 .formulas.py`)
# ---------------------------------------------------------------------------

def _self_test() -> int:
    cases: List[Tuple[str, str, int]] = [
        ("x**2 + sin(a)*b", "infix", 3),
        (r"\sin(a) + \frac{b}{2}", "latex", 2),
        (r"\sum_{i=0}^{k} i**2", "latex", 1),  # i is bound by sum
        (r"\sqrt{x^2 + y**2}", "latex", 2),
    ]

    failures = 0
    for text, mode, expected_free in cases:
        try:
            f = Formula.parse(text, mode=mode)
        except Exception as exc:
            print(f"FAIL {text!r}: {exc}")
            failures += 1
            continue

        free_count = len(f.expr.free_symbols)
        if free_count != expected_free:
            print(f"FAIL {text!r}: expected {expected_free} free symbols, got {free_count}")
            failures += 1
        else:
            print(f"OK   {text!r}: free={sorted(s.name for s in f.expr.free_symbols)}")
            print(f"      ascii=\n{f.ascii()}")

    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_self_test())
