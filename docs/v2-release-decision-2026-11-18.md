# Football Copilot V2 — bounded product release decision (18 November 2026)

## Decision

**GO: retain the integrated facts-first V2 as a usable private preview.**
**HOLD / NO-GO: do not promote it to production yet.**

Stop the V2 architectural research loop. No more tool-selection experiments, no
new framework, and no incremental work solely to increase the test count.

## Product consolidation

The private app at `app/v2_groq_preview.py` uses
`agent/product_conversation_v2.py` for recognised Premier League recent-form
questions. Existing verified Python and DuckDB tools handle historical form,
points, scoring, defence, home/away splits, opponent context, two-window trends,
H2H, and session follow-ups. The existing legacy V2 route remains available only
when explicitly configured. The exploratory avenue-selection and investigator
research modules are **not promoted** or added to this product route.

Product chat shows a concise conclusion, expandable verified statistics and
sources, and follow-up suggestions only beneath the latest answer. The user
reviewed three live Arsenal/Liverpool questions; targeted fixes improved venue
headlines and broad form summaries. A final real failure, a follow-up containing
"defensively" inheriting the earlier attacking focus, was corrected.

## Evidence — bounded 20-case screen

Run: `python -m scripts.assess_v2_product_release`

- Integrated V2: **20/20** scored cases
- Existing V2 **offline deterministic-only** route: **9/20**
- Full Python regression: **179/179** passing
- Five sessions, 20 authored question turns, covering general comparisons,
  attacking/defensive follow-ups, venue rates, opponent context, H2H,
  club switching, five-versus-five trends and an unsupported question.
- The benchmark checks expected tools, teams, windows and basic relevance.
  It is **not** a statistically representative natural-language comprehension
  study or a blinded human judgement of explanations.
- The prior V2 baseline deliberately disables remote model planning. Its
  results **must not** be presented as a win over full Groq-enabled V2 or
  the original deployed production application.
- Some GitHub PR merge checkouts have different match-data snapshots from the
  private preview. Numbers must be sourced from the current checkout; no
  hardcoded observed goals or results were used as success thresholds.
- No new evidence of gains against the separate 1X2 accuracy hurdle
  (54.00% established fixed 40/60 benchmark; target >=56.00%).

## Release gate

1. User reviews the refreshed **private** port-8501 build and confirms that
   answers are understandable, useful and correct for unscripted questions.
2. Resolve any material blocker identified by that review only.
3. Production remains unchanged unless that review establishes enough benefit
   and the user explicitly approves promotion.

This is the end of the bounded V2 development-and-evaluation cycle; future
research requires a newly stated, measurable product limitation.
