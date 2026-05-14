# Learning Resources for Wyatt

## Week 1 — Foundations

- **Flower quickstart tutorial:** https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
- **Federated Learning explained (visual):** https://ai.googleblog.com/2017/04/federated-learning-collaborative.html
- **What is VFL? (3-min video):** Search YouTube "vertical federated learning explained"

## Week 2 — Differential Privacy

- **Opacus quickstart (MUST COMPLETE before coding):** https://opacus.ai/tutorials/
  - Run the MNIST DP example — target: training accuracy > 95% with epsilon=10
- **"A friendly introduction to differential privacy" (blog):**
  https://desfontain.es/privacy/friendly-intro-to-differential-privacy.html
- **Dr. Li's reading guide:** `docs/dp_reading_guide.pdf` ← ask Dr. Li for this file

## Week 2–3 — PSI / Cryptography

- **Python `cryptography` library ECDH example:**
  https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ec/
- **PSI explained in plain English:**
  https://www.wikiwand.com/en/Private_set_intersection

## Week 7–8 — Attacks

- **iDLG paper with code (PyTorch):** https://github.com/PatrickZH/Improved-Deep-Leakage-from-Gradients
  - Study the `iDLG.py` file — Algorithm 1 is implemented there
- **CAFE attack code:** https://github.com/Zhuokai-Zhao/CAFE

## Common Error Solutions

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: opacus` | Package not installed | `pip install opacus==1.4.1` |
| `BatchNorm not supported in Opacus` | Opacus can't handle BN | Tell Alex — he will replace with GroupNorm |
| `ConnectionRefusedError` in PSI test | Server not started | Run `python -m psi.psi_server` first in a separate terminal |
| `FileNotFoundError: silo_a.pkl` | Silos not prepared | Run `python scripts/prepare_silos.py` |
| `PrivacyBudgetExceededError` | Training exceeded epsilon budget | Reduce `num_epochs` or increase `epsilon` in config |
