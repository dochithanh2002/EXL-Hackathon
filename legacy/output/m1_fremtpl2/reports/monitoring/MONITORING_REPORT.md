# Monitoring Report: M1

_Generated 2026-11-13T16:52:27+00:00. Champion `output/m1_fremtpl2/model/m1_fremtpl2_freq.py` (sha256 `fda45c0fbca2`). Baseline = learning sample; validated test loss 30.9109._

**Re-validation required: YES**

| Metric | Amber / red | test cohort | young-driver growth |
|---|---|---|---|
| freq_psi | 0.1 / 0.25 | 🟢 0.0001 | 🟢 0.0147 |
| max_csi | 0.1 / 0.25 | 🟢 0.0004 | 🟠 0.1004 |
| ae_dev | 0.05 / 0.1 | 🟢 0.0315 | 🟢 0.0403 |
| loss_increase_rel | 0.02 / 0.05 | 🟢 0.0000 | 🟢 0.0189 |
| max_age_band_ae_dev | 0.1 / 0.2 | 🟢 0.0700 | 🔴 0.2631 |
| Overall | | 🟢 **GREEN** | 🔴 **RED** |

## test cohort: GREEN

Simulated batch: held-out policies with observed claims.

- No breaches.

## young-driver growth: RED

Simulated scenario: test policies with drivers aged 18-25 tripled (portfolio-mix shift); observed claims of those policies reused.

- max_csi = 0.1004 (AMBER; amber/red at 0.1/0.25), driven by DrivAgeGLM
- max_age_band_ae_dev = 0.2631 (RED; amber/red at 0.1/0.2)

