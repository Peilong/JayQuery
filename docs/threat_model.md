# Threat Model

> Owner: Wyatt Wilfong | Due: Week 7 | Status: TEMPLATE

## Attacker Identity
**Honest-but-curious server**: The FL aggregation server follows the protocol correctly
(it does not deviate or crash), but it *also* tries to learn as much as possible about
the raw student data from the information it legitimately observes.

## Attacker Capabilities
The attacker (server) can observe:
- [ ] Embedding vectors (emb_a, emb_b) received from both silos each round
- [ ] The gradient tensors (dL/d_emb_a, dL/d_emb_b) that it computes and returns
- [ ] The aggregate loss value each round
- [ ] The Top Model weights

The attacker CANNOT observe:
- [ ] Raw features in Silo A (imd_band, demographics)
- [ ] Raw features in Silo B (VLE clicks, assessment scores)
- [ ] Student identifiers (PSI ensures the server only learns the intersection size)
- [ ] Student labels (Silo B holds labels privately)

## Attack Objective
Primary: Reconstruct `imd_band` (deprivation index) for individual students from emb_a.
Secondary: Infer at-risk labels from the gradient direction returned to Silo B.

## Attack Method
**Model Inversion (iDLG, Zhao 2020)**: Optimize a dummy input to minimize the L2 distance
between its embedding and the real embedding. Measure success as Attack Success Rate (ASR).

## Defense
**Differential Privacy (DP-SGD, Abadi 2016)**: Inject Gaussian noise into gradients before
transmission to the server. Formal guarantee: (epsilon, delta)-DP.

## Success Metric
At epsilon=1.0, ASR should be ≤ 1/10 (random for 10-class imd_band, ≈10%).
At epsilon=inf (no DP), ASR should be meaningfully > 10%.

> Fill in actual ASR values from experiments in Week 9.
