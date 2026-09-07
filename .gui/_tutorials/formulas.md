# <img class='emoji_nav' src='emojis/crystal_ball.svg' /> The `--formula` parameter: optimizing equations directly

<!-- Describe a math formula (LaTeX / infix) instead of writing a run-program -->

<!-- Category: Preparations, Basics and Setup -->

<div id="toc"></div>

## What is `--formula`?

Normally OmniOpt2 optimizes a black-box program: you write a `run.sh` (or any executable) that prints a `RESULT: …` line, OmniOpt2 sweeps it with different hyperparameter values. The `--formula` flag lets you skip that step and **describe the objective as a math formula instead**. OmniOpt2 parses the formula with [sympy](https://www.sympy.org/) (so sums, products, integrals, derivatives and limits all work), infers the free symbols and writes a tiny Python runner that evaluates the formula for each trial.

This is useful when:

- You want to test OmniOpt2 on a known mathematical landscape before plugging in a real program.
- You have an analytical objective (loss surface, toy benchmark, kernel, …) and want to sweep it.
- You want a quick sanity check that a search space is well-shaped before investing in a full run.

## Quick start (CLI)

A typical invocation looks like this (taken from the same set of flags a normal `.tests/` run would use):

```bash
./omniopt --partition=alpha --experiment_name=a --mem_gb=10 --time=60 --worker_timeout=60 \
    --max_eval=500 --num_parallel_jobs=20 --gpus=1 --num_random_steps=20 --follow \
    --send_anonymized_usage_stats --result_names 'RESULT=min' --cpus_per_task=1 \
    --nodes_per_job=1 --revert_to_random_when_seemingly_exhausted \
    --model=BOTORCH_MODULAR --n_estimators_randomforest=100 --optuna_pruner=none \
    --optuna_n_startup_trials=10 --optuna_n_ei_candidates=0 \
    --optuna_study_name=omniopt_study --optuna_extra_iters=1 --run_mode=local \
    --occ_type=euclid --main_process_gb=8 --nr_evals_per_arm=1 \
    --max_nr_of_zero_results=50 --slurm_signal_delay_s=0 --max_failed_jobs=0 \
    --max_attempts_for_generation=20 --num_restarts=20 --raw_samples=1024 \
    --max_abandoned_retrial=20 --max_num_of_parallel_sruns=16 \
    --number_of_generators=1 --generate_all_jobs_at_once \
    --formula="$(printf '%s' 'f(a,b) = a - b' | base64 -w0)" \
    --parameter a range -1000 1000 float false \
    --parameter b range -1000 1000 float false
```

When OmniOpt2 sees `--formula` (and no `--run_program`):

<ol>
<li>It decodes the formula from base64.</li>
<li>It parses the formula.</li>
<li>It writes a tiny <code>run_with_formula.py</code> helper into the run folder.</li>
<li>It replaces the (missing) <code>--run_program</code> with a call to that helper that exports each parameter as an env var <code>OMNIOPT_PARAM_&lt;name&gt;=…</code> and prints <code>RESULT: …</code> per trial.</li>
</ol>

That's it — there is no automatic detection of free symbols on the CLI. The example above explicitly passes `--parameter a range -1000 1000 float false` and `--parameter b range -1000 1000 float false`; without those, OmniOpt2 has nothing to sweep. The GUI is what suggests parameters for you; on the CLI you write them yourself.

You don't have to base64-encode by hand on the CLI — the GUI does it for you — but on the CLI it's the safest way to get backslashes, spaces and quotes through bash. Both `--formula 'f(a,b) = a - b'` (raw) and `--formula="$(… | base64 -w0)"` (encoded) are accepted.

You will see something like this in the log (the last line is just informational — the free symbols are detected from the parsed formula, not from `--parameter`):

```
[Formula]
a - b
[Formula LaTeX] a - b
[Formula hyperparameters] a, b
```

## Quick start (GUI)

In the GUI, the **Run program** field has a small tab bar underneath it. Click the **Formula** tab and a math editor appears: a text input on the left with a live MathJax preview underneath, and a **Suggested parameters** panel on the right with an **Apply → add as parameters** button. Toggling the **Formula mode** select (Auto / LaTeX / Infix) switches the parser on the fly.

When you click **Apply**, the suggested hyperparameters are pushed into the main parameter table and the **Run program** textarea is replaced by the auto-generated helper. The two are mutually exclusive — exactly one of the two must be filled. If you want to go back to a regular script, just click **Clear formula** and the **Run program** tab comes back.

## Modes: `--formula_mode`

OmniOpt2 accepts three modes for parsing:

<table>
<tr class="invert_in_dark_mode">
	<th>Mode</th>
	<th>When to use</th>
	</tr>
	<tr>
	<td>
	<code>auto</code> (default)</td>
	<td>Picks <code>latex</code> if the formula contains a backslash, otherwise <code>infix</code>. Best for general use.</td>
	</tr>
	<tr>
	<td>
	<code>latex</code>
	</td>
	<td>Forces LaTeX-style input. Use this when <code>auto</code> mis-detects (e.g. an infix expression with backslashes in string literals).</td>
	</tr>
	<tr>
	<td>
	<code>infix</code>
	</td>
	<td>Forces Python-style infix. Use this when the formula is pure Python syntax without any LaTeX.</td>
	</tr>
</table>

Pick the mode in the GUI via the **Formula mode** select next to the editor, or on the CLI:

```bash
--formula_mode=latex
--formula_mode=infix
--formula_mode=auto   # default
```

If parsing fails, OmniOpt2 prints the underlying sympy error. Common causes:

- Empty `\frac{}{}`, `\int^{}_{}` or unterminated `\sum_{`.
- An `=` inside a brace block that was not intended as an assignment.
- A stray backslash followed by a space (`\ sin`).

## Supported LaTeX

The preprocessor in `.formulas.py` understands a useful subset of LaTeX. Anything beyond that is passed through to sympy's `parse_latex` (when `antlr4-python3-runtime` is installed) or to `parse_expr` as a last resort.

### Operators

<table>
<tr class="invert_in_dark_mode">
	<th>LaTeX</th>
	<th>Meaning</th>
	</tr>
	<tr>
	<td>
	<code>+</code>, <code>-</code>, <code>*</code>, <code>/</code>
	</td>
	<td>usual arithmetic</td>
	</tr>
	<tr>
	<td>
	<code>^</code> and the double-star operator</td>
	<td>power (both accepted)</td>
	</tr>
	<tr>
	<td>
	<code>\cdot</code>, <code>\times</code>, <code>\ast</code>
	</td>
	<td>multiplication</td>
	</tr>
	<tr>
	<td>
	<code>\frac{a}{b}</code>, <code>\dfrac{a}{b}</code>, <code>\tfrac{a}{b}</code>
	</td>
	<td>division <code>a/b</code>
	</td>
	</tr>
	<tr>
	<td>
	<code>\sqrt{x}</code>, <code>\sqrt[n]{x}</code>
	</td>
	<td>square / n-th root</td>
	</tr>
	<tr>
	<td>
	<code>|x|</code>
	</td>
	<td>absolute value (also <code>\|x\|</code>)</td>
	</tr>
</table>

### Functions

<table>
<tr class="invert_in_dark_mode">
	<th>LaTeX</th>
	<th>Sympy</th>
	</tr>
	<tr>
	<td>
	<code>\sin</code>, <code>\cos</code>, <code>\tan</code>
	</td>
	<td>
	<code>sin</code>, <code>cos</code>, <code>tan</code>
	</td>
	</tr>
	<tr>
	<td>
	<code>\asin</code>, <code>\acos</code>, <code>\atan</code>
	</td>
	<td>
	<code>asin</code>, <code>acos</code>, <code>atan</code>
	</td>
	</tr>
	<tr>
	<td>
	<code>\sinh</code>, <code>\cosh</code>, <code>\tanh</code>
	</td>
	<td>
	<code>sinh</code>, <code>cosh</code>, <code>tanh</code>
	</td>
	</tr>
	<tr>
	<td>
	<code>\exp</code>, <code>\log</code>, <code>\ln</code>
	</td>
	<td>
	<code>exp</code>, <code>log</code>, <code>ln</code>
	</td>
	</tr>
	<tr>
	<td>
	<code>\abs{x}</code>
	</td>
	<td>
	<code>Abs(x)</code>
	</td>
	</tr>
</table>

### Sums, products, integrals and limits

<table>
<tr class="invert_in_dark_mode">
	<th>LaTeX</th>
	<th>Meaning</th>
	</tr>
	<tr>
	<td>
	<code>\sum_{i=0}^{n} f</code>
	</td>
	<td>
	<code>Sum(f, (i, 0, n))</code>
	</td>
	</tr>
	<tr>
	<td>
	<code>\sum_{i} f</code>
	</td>
	<td>defaults to <code>(i, 0, oo)</code>
	</td>
	</tr>
	<tr>
	<td>
	<code>\prod_{i=0}^{n} f</code>
	</td>
	<td>
	<code>Product(f, (i, 0, n))</code>
	</td>
	</tr>
	<tr>
	<td>
	<code>\int_{a}^{b} f \,dx</code>
	</td>
	<td>
	<code>Integral(f, (x, a, b))</code>
	</td>
	</tr>
	<tr>
	<td>
	<code>\int f \,dx</code>
	</td>
	<td>defaults to <code>(-oo, oo)</code>
	</td>
	</tr>
	<tr>
	<td>
	<code>\lim_{x \to v} f</code>
	</td>
	<td>
	<code>Limit(f, x, v)</code>
	</td>
	</tr>
	<tr>
	<td>
	<code>\frac{d}{dx} f</code>
	</td>
	<td>
	<code>Derivative(f, x)</code>
	</td>
	</tr>
</table>

The bound variable (`i`, `j`, `k`, …) is automatically excluded from the hyperparameter list.

### Macros that are stripped (formatting only)

`\text{…}`, `\textit{…}`, `\mathrm{…}`, `\mathbf{…}`, `\mathcal{…}`, `\mathbb{…}`, `\mathfrak{…}`, `\mathsf{…}`, `\mathtt{…}`, `\operatorname{…}`, `\mbox{…}`, `\boldsymbol{…}`, `\hat`, `\tilde`, `\bar`, `\vec`, `\dot`, `\widehat`, `\widetilde`, `\overbrace`, `\underbrace`, `\overline`, `\underline`, `\left`, `\right`, `\,`, `\;`, `\!`, `\quad`, `\qquad`, `\ `, `\displaystyle`.

These don't change the math — they exist purely for typesetting.

### Greek letters and special constants

`\alpha` → `alpha`, `\beta` → `beta`, …, `\omega` → `omega` (same names in both upper and lower case). `\pi` → sympy's `pi`, `\infty` → `oo`. Bare `e` and `E` are treated as Euler's number when they appear in the body and are **not** turned into hyperparameters.

## Supported infix

Infix mode accepts plain Python expressions that sympy's `parse_expr` can chew on. Highlights:

```python
sin(x) + cos(y)                    # trig
exp(-x**2) / sqrt(2*pi)            # gaussian
Sum(i**2, (i, 0, n))               # sum with bound variable
Product((x - i), (i, 1, k))        # product
Integral(exp(-x**2), (x, -oo, oo)) # integral
Derivative(f, x)                   # derivative
Limit(sin(x)/x, x, 0)              # limit
Min(a, b)  Max(a, b)               # min / max
abs(x - y)                         # absolute value
sign(x)  floor(x)  ceil(x)         # rounding
2**x  x**2  x**y                   # powers
```

Implicit multiplication is on by default: `2x`, `xy`, `(a)(b)` all work.

## Worked examples

Every example below is exercised by an automated test in `.tests/test_formulas` (the `test_tutorial_*` functions). If a tutorial example starts failing, that test will fail too.

### Sphere (quick start)

```bash
--formula="$(printf '%s' 'f(x, y) = x**2 + y**2' | base64 -w0)" \
--parameter x range -1000 1000 float false \
--parameter y range -1000 1000 float false
```

### Quadratic loss

```bash
--formula="$(printf '%s' 'f(x) = a*x**2 + b*x + c' | base64 -w0)" \
--parameter x range -1000 1000 float false
```

`a`, `b` and `c` appear only on the RHS. The GUI would classify them as constants; the CLI however treats every free symbol as a range parameter, so without explicit `--parameter a/b/c …` lines OmniOpt2 would auto-fill them with `range -1 1 float false` and try to optimize them too. The example above only lists `x` — that means OmniOpt2 will auto-fill `a`, `b`, `c` as additional `range -1 1 float` parameters (look for the `[Formula] Auto-filled --parameter: …` log line). If you actually want `a`, `b`, `c` to stay at a fixed value, pass them as `--parameter a fixed 3` etc. explicitly. Variables whose name ends in the suffix `_int` are suggested as `int`; here everything is `float`.

### Sigmoid

```bash
--formula="$(printf '%s' '\sigma(z) = \frac{1}{1 + e^{-z}}' | base64 -w0)" \
--formula_mode=latex \
--parameter z range -10 10 float false
```

### Polynomial kernel

```bash
--formula="$(printf '%s' 'K(x, y) = (x \cdot y + c)^{d}' | base64 -w0)" \
--formula_mode=latex \
--parameter x range -10 10 float false \
--parameter y range -10 10 float false
```

### Sigmoid cross-entropy

```bash
--formula="$(printf '%s' 'L = -1/n * Sum(y_i * log(sigmoid(w*x_i + b)) + (1 - y_i) * log(1 - sigmoid(w*x_i + b)), (i, 0, n))' | base64 -w0)" \
--parameter w range -5 5 float false \
--parameter b range -5 5 float false \
--parameter n range 1 100 int false
```

### Definite integral

```bash
--formula="$(printf '%s' '\int_{0}^{1} x**2 \,dx' | base64 -w0)"
```

Note: a definite integral with constant bounds evaluates to a constant — there are no free symbols at all here, so the helper script has nothing to sweep and the run exits immediately. To get something to optimise, make at least one bound a hyperparameter, e.g. `\int_{0}^{n} x^2 \,dx` with `--parameter n range 0 10 float false`.

### Bare infix

```bash
--formula="$(printf '%s' 'sin(x)**2 + cos(x)**2' | base64 -w0)" \
--formula_mode=infix \
--parameter x range 0 6.283185 float false
```

This is the trigonometric identity `sin² + cos² = 1` and evaluates to `1` for every `x`, so OmniOpt2 should report a flat objective.

### Multi-objective

A formula whose body is a tuple `(… , …)` is automatically treated as multi-objective. OmniOpt2 prints one `RESULT_…` line per tuple component and matches them up against `--result_names` (one entry per component, in order). Without `--result_names` everything is minimized by default.

```bash
--formula="$(printf '%s' 'f(x) = (x**2, (x - 3)**2)' | base64 -w0)" \
--result_names='OBJ1=min OBJ2=min' \
--parameter x range -10 10 float false
```

## How parameters are inferred

The GUI and the CLI use slightly different but equivalent heuristics. Both look at the **left-hand side** of an optional `=` assignment first:

<table>
<tr class="invert_in_dark_mode">
	<th>Formula</th>
	<th>Parameters</th>
	<th>Constants</th>
	</tr>
	<tr>
	<td>
	<code>f(x, y) = x^2 + y^2</code>
	</td>
	<td>
	<code>x</code>, <code>y</code>
	</td>
	<td>—</td>
	</tr>
	<tr>
	<td>
	<code>f(x) = a*x + b</code>
	</td>
	<td>
	<code>x</code>
	</td>
	<td>
	<code>a</code>, <code>b</code> (RHS-only)</td>
	</tr>
	<tr>
	<td>
	<code>f(x) = \sum_{i=0}^{n} i*x</code>
	</td>
	<td>
	<code>x</code>
	</td>
	<td>
	<code>n</code> (RHS-only; <code>i</code> is bound, not suggested)</td>
	</tr>
	<tr>
	<td>
	<code>a + sin(b)</code> (no LHS)</td>
	<td>
	<code>a</code>, <code>b</code>
	</td>
	<td>— (RHS-only fallback treats everything as a parameter)</td>
	</tr>
</table>

Once the LHS is stripped, the parser classifies the remaining free symbols:

- Anything on the LHS is a **parameter** (range by default).
- Anything that only appears on the RHS is a **constant** (fixed to its default value, but you can flip it to a range in the GUI).
- Variables bound by `\sum`, `\prod`, `\int`, `\lim` or `\frac{d}{dx}` are excluded.
- The literal `e`, `E`, `pi`, `PI` are excluded and substituted with their conventional values.

If you want to suggest a default range other than `[-1, 1]`, name the variable accordingly (this only matters in the GUI's auto-suggest table — on the CLI you always pass `--parameter` yourself):

<table>
<tr class="invert_in_dark_mode">
	<th>Naming hint</th>
	<th>Effect</th>
	</tr>
	<tr>
	<td>name ending in <code>_int</code>
	</td>
	<td>suggested as <code>int</code> with <code>[0, 10]</code>
	</td>
	</tr>
	<tr>
	<td>name starting with <code>lr_</code> or <code>log_</code>, or ending in <code>_log</code>
	</td>
	<td>suggested with <code>log_scale=true</code> and <code>[1e-5, 1e-1]</code>
	</td>
	</tr>
	<tr>
	<td>anything else</td>
	<td>
	<code>[-1, 1]</code>, <code>float</code>, <code>log_scale=false</code>
	</td>
	</tr>
</table>

For example, a variable named `lr_learning_rate` or `epochs_int` is auto-tuned correctly without any further editing.

## How `--formula` works internally

Behind the scenes, OmniOpt2 generates a tiny per-run helper called `run_with_formula.py` in the run folder, hands the parsed formula to it, and uses that helper as the `--run_program` for every trial. The helper reads each trial's hyperparameter values from environment variables (`OMNIOPT_PARAM_<name>=…`), evaluates the formula with [sympy](https://www.sympy.org/), and prints a single `RESULT: …` line (or one `RESULT_0`, `RESULT_1`, … line per component for the multi-objective case). You normally never look at or edit this script — it's just the mechanism OmniOpt2 uses to turn a math equation into a `run_program`.

## What gets written to the run folder

A formula-based run drops four extra files (`formula.txt`, `formula_pretty.txt`, `formula_underbraces.txt`, `formula_params.json`) plus the auto-generated `run_with_formula.py` into the run folder. See the [folder structure tutorial](tutorials?tutorial=folder_structure) for what each one contains.

## Continuing a formula-based job

`--continue_previous_job` works as usual: the next run re-uses the previous run folder, picks up `formula.txt` / `formula_params.json` and re-uses the existing `run_with_formula.py`. No need to re-supply `--formula` (and no need to worry about shell-quoting it again).

```bash
omniopt --continue_previous_job=runs/my_experiment/42 … --follow
```

If you do want to override the formula on a continued run, just pass `--formula` (and `--formula_mode` if needed); the new value wins.

## Sharing a formula-based run

`omniopt_share` (or the **Share** button in the GUI) automatically picks up the four formula files and renders the formula with clickable parameter overlays on the share page. Nothing extra to do.

## Interaction with other options

- `--run_program` and `--formula` are **mutually exclusive**. If both are present, the explicit `--run_program` wins and the formula is silently dropped. If neither is present and `--continue_previous_job` is unset, OmniOpt2 exits with code 19 ("--run_program was empty").
- `--parameter` is **optional in principle, required in practice**. If you omit it, OmniOpt2 auto-fills it from the formula's suggestions (with `range` or `fixed` and the unhelpful `[-1, 1]` default). Always pass `--parameter` explicitly with ranges that match your problem — see the [Quick start (CLI)](#quick-start-cli) section above.
- `--formula_python_path` lets you point OmniOpt2 at a specific Python interpreter (defaults to the current `sys.executable`). Useful when the auto-detected interpreter doesn't have sympy installed.
- `--formula_mode` and the GUI's **Formula mode** select map directly onto the same parser dispatch.
- All other options (`--model`, `--num_parallel_jobs`, `--max_eval`, `--constraint …`, etc.) work the same as for a regular run.

## Troubleshooting

<table>
<tr class="invert_in_dark_mode">
	<th>Symptom</th>
	<th>Likely cause</th>
	</tr>
	<tr>
	<td>
	<code>Could not parse --formula …: …</code>
	</td>
	<td>Empty/unbalanced LaTeX group, stray <code>\</code>, or unsupported syntax. Fix the formula and retry, or supply <code>--run_program</code> as a fallback.</td>
	</tr>
	<tr>
	<td>The GUI shows "Preview error: unbalanced braces"</td>
	<td>A <code>{</code> without a matching <code>}</code> (or vice-versa).</td>
	</tr>
	<tr>
	<td>Suggested parameters don't include a variable</td>
	<td>It's bound by <code>\sum</code> / <code>\prod</code> / <code>\int</code>, or it's a reserved name (<code>sin</code>, <code>pi</code>, <code>oo</code>, …), or it's <code>e</code> / <code>pi</code> / <code>E</code> (treated as a fixed constant).</td>
	</tr>
	<tr>
	<td>Auto-suggested range is <code>[-1, 1]</code> when you wanted <code>[0, 1]</code>
	</td>
	<td>Name the variable <code>lr_…</code>, <code>log_…</code> or <code>…_log</code> for log scale; otherwise set the range manually in the GUI.</td>
	</tr>
	<tr>
	<td>The CLI call works but the GUI's curl fails</td>
	<td>The base64 round-trip through <code>printf '%s' … | base64 -w0</code> was missing — the GUI sends base64 and the shell needs to decode it. Use <code>$(printf '%s' '…' | base64 -w0)</code> exactly.</td>
	</tr>
	<tr>
	<td>
	<code>OMNIOPT_PARAM_x</code> is empty / <code>nan</code>
	</td>
	<td>The parameter name was renamed (e.g. <code>\lambda</code> → <code>lam_</code>) by the LaTeX preprocessor. Use the renamed name in your formula.</td>
	</tr>
</table>

For unit-level coverage of the parser, suggestions and lambdify see `.tests/test_formulas`. The CLI self-test `python3 .formulas.py` exercises a handful of representative cases from the command line. Every example in the **Worked examples** section above has its own `test_tutorial_*` entry in `test_formulas`, so a regression in any documented example will fail the smoke tests in CI.
