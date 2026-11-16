# R-Pythonizer — Demo video script v2 (for review)

Hard limit 3:00 (EXL submission rules, slide 13). Planned length ≈ 2:50 (VO ≈ 100 s in Part A at the v1 pace).
Sound: v1 music bed. Tone: a student team pitching an idea it is excited about.
Part A (context → platform) ≈ 1:56, Part B (live demo on M4) ≈ 0:55.
Every number on screen has a source in the table at the end.

---

## Part A — Context, approach, solution, platform

### A1 · Hook: the model that lives on a laptop — 0:03–0:16
**Visual:** a laptop on cream. Window `credit_score.R · last edited 2014`. An email pops up:
`Re: Q3 scores → attached.xlsx`. Then a badge on the author's avatar: `left the bank`. Cut to a big counter
`20 years` with the line `R: the statisticians' language`.
**VO:**
> For twenty years, risk models were built in R. A statistician writes the script, runs it on their own laptop,
> and emails the numbers. It works, until that person leaves.

### A2 · Why move to Python — 0:16–0:28
**Visual:** three limit chips appear on the laptop: `runs in memory` · `one thread at a time` · `hard to plug into
production`. Then two bars: `Python 57.9%` (orange, long) vs `R 4.9%` (navy, short), label `developers using it, 2025`.
**VO:**
> R runs in memory, one thread at a time, and it's hard to plug into production. Meanwhile, fifty-eight percent of
> developers use Python. R? Five.

### A3 · Why R, not SAS — 0:28–0:42
**Visual:** two columns. Left `SAS`: `one vendor` · `migration already industrialised` · `EXL Code Harbor:
−70% conversion time`. Right `R` (orange): `22,800+ open packages` · `each with its own defaults` · `no tool proves
the result`. End beat: `R is the gap.`
**VO:**
> So why R, and not SAS? SAS migration is already industrialised: EXL's own Code Harbor cut conversion time by
> seventy percent. R is the gap: twenty-two thousand open packages, each with its own defaults.

### A4 · Problem: defaults are the trap — 0:42–0:57
**Visual:** reuse of the v1 A2 visual, shortened: R and Python panels, chip `scikit-learn adds a penalty`, chip
`pandas reads more values as missing`, coefficient `−0.41237 → −0.41249` (illustrative),
`APPROVE → DECLINE`. Pill on top: `PRA SS1/23 · a rewrite is a model change`.
**VO:**
> And those defaults are the trap. Under PRA SS1/23, a rewrite is a model change that needs evidence. One hidden
> penalty or missing-value rule, and a coefficient shifts enough to flip a loan decision.

### A5 · Meet R-Pythonizer, and its approach — 0:57–1:16
**Visual:** first beat: logo mark + `R-Pythonizer` + subtitle `A web platform that moves statistical models from R
to Python, with proof`. Then `AI writes` · `Code proves` · `Human approves`, one per beat. Scope strip:
`Classical statistical models → statsmodels` · `Proven: binomial GLM, Poisson GLM` · `Unsupported → refused,
never substituted`.
**VO:**
> Meet R-Pythonizer: a web platform that moves statistical models from R to Python, and proves the result. The
> rule is simple. AI writes the Python. Plain code proves it. A human approves. And if a model has no faithful
> Python equivalent, the platform says so and stops.

### A6 · The proof — 1:10–1:29
**Visual:** v1 A4 pipeline: `baseline.R` → `R answer key` → `R's exact rows` → `Python candidate` → `Parity
check`, lock on the Translator `sees names, never values`, formula `|Python − R| ≤ atol + rtol·|R|`, then the
repair loop.
**VO:**
> R runs the model once and freezes an answer key. Python gets R's exact rows, and the AI never sees R's numbers,
> so it can't cheat. Then plain code checks every coefficient and prediction, and if anything fails, an agent
> finds out why.

### A7 · Beyond parity — 1:29–1:39
**Visual:** chips `Calibration` `AUC / Gini` `Stability (PSI)` `Fairness` `Assumptions` `Sample size`;
`Migration accepted ≠ Fit for use`.
**VO:**
> But a perfect copy of a bad model is still a bad model, so we also check calibration, stability and fairness,
> as a separate verdict.

### A8 · Platform and benchmark — 1:39–1:56
**Visual:** seven-step stepper and four agent cards, then the benchmark chart from the last render: GPT rows
highlighted, gpt-6.1-terra tagged `fastest`, cost and time columns, headline `Fast and cheap. Same verified
result.`
**VO:**
> Seven fixed steps, four narrow agents, zero data rows, and a script anyone can re-run. We tried four AI models,
> and GPT-6.1 terra, fast and cheap, got the same verified result for under five cents.

---

## Part B — Live demo on M4 — 1:56–2:51

Unchanged from the last render (real M4 screens, real numbers).

### B0 · Card — `Live demo · M4 · GiveMeSomeCredit` (3 s, no VO)

### B1 · Data & Code
> Let's try it on a real credit-default model: an R notebook and a hundred and fifty thousand rows.

### B2 · Model Specification
> The Analyst agent drafts the spec, and even flags what's missing, like a labelled holdout set.

### B3 · R Reference Results
> R runs once and freezes the answer key.

### B4 · Python Migration & Comparison
> The Translator writes Python, and twenty-five checks pass: coefficients and predictions match R to thirteen
> decimal places. Business fitness? Not established yet, and the platform tells you exactly why.

### B5 · Validation Report & Package
> The report backs every number with evidence, and the package re-runs the proof anywhere.

### B6 · Close
> R-Pythonizer. AI writes, code proves, humans approve.

### TEAM · The Fantastic 4 (5 s, no VO)

---

## Sources for on-screen numbers

| Claim | Source |
|---|---|
| Python used by 57.9% of developers, R by 4.9% (49,000+ respondents) | [Stack Overflow Developer Survey 2025](https://survey.stackoverflow.co/2025/technology) |
| ~22,800 active CRAN packages (30 Sep 2025) | [R Journal, Changes on CRAN 2025-3](https://journal.r-project.org/news/RJ-2025-3-cran/) |
| EXL Code Harbor: SAS → Python for a UK financial institution, 70% less conversion time | [EXL case study](https://www.exlservice.com/insights/case-study/exl-transforms-code-migration-for-leading-uk-financial-institution) |
| Other SAS → Python converters exist (Hexaware Amaze, Indium Data Lifter, LTIMindtree Scintilla) | [Hexaware](https://hexaware.com/flyer/automate-sas-to-pyspark-conversion-shift-to-python-faster-using-genai/), [Indium](https://www.indium.tech/?p=41138) |
| R is single-threaded by default and works in memory | [RStudio: R for Enterprise](https://rviews.rstudio.com/2016/12/21/r-for-enterprise-how-to-scale-your-analytics-using-r/), [R in production](https://www.r-bloggers.com/2023/06/lessons-learned-from-running-r-in-production/amp/) |
| R was the statisticians' language for 15–20 years; many models written by people who have moved on | `TEAM_BRIEFING.md` §1.2 |
| A rewrite is a model change under PRA SS1/23 | `TEAM_BRIEFING.md` §1.5 |
| scikit-learn penalty, pandas missing values | `TEAM_BRIEFING.md` §1.4 |
| Benchmark: four AI models, costs, times | `TEAM_BRIEFING.md` M4 benchmark |
| M4 demo numbers | live project M4 on the platform |

The laptop-and-email scene in A1 is an illustration of a common pattern, not a specific bank.
