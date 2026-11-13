# Human Review: M2 migration

Approve in Claude Code with `/mrf:approve m2_taiwan` (or `agent/bin/mrf approve m2_taiwan --target migration --reviewer "<name>" --decision approve|reject --comment "..."`).

Reviewer checklist:
- [ ] Read `m2_taiwan_pd.py`. Every R preprocessing step is present, in R order (report §2).
- [ ] Positional references were resolved to the correct column names (R-IDX-01).
- [ ] Factor reference levels and dummy names match R (`reference/meta.json` → xlevels).
- [ ] The split is imported from R, not re-randomised (R-RNG-01).
- [ ] Parity evidence (§5) is within the frozen tolerances, and `.venv/bin/python -m pytest output/m2_taiwan/model/tests` passes.
- [ ] Risks and protected attributes (§2) are handed to independent validation. Migration does NOT fix them (GOV-01).
- [ ] Manifest hashes match the files under review.
