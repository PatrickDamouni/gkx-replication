# Decision log

One entry per decision: what I chose, what I rejected, why. Newest at the bottom.
This file is the interview script.

---

## D001 — Project choice: replicate GKX (2020) instead of a novel model
- **Chose:** Replicate + extend Gu, Kelly & Xiu, "Empirical Asset Pricing via Machine Learning."
- **Rejected:** Building an original trading model; benchmarking TS foundation models; LLMs on filings.
- **Why:** A replication has a ground truth to check against, so mistakes are detectable.
  Not being a PhD student, the strongest signal I can send is evaluation rigor, and a
  replication puts that front and center.

## D002 — Data: CRSP via MSU WRDS + Open Source Asset Pricing signals
- **Chose:** CRSP monthly returns (WRDS, MSU access) merged with OSAP firm-level characteristics.
- **Rejected:** yfinance (survivorship bias — delisted firms drop out); OSAP portfolio-level
  returns only (can't do firm-level return prediction, which is the paper's core task).
- **Why:** Firm-level, survivorship-free returns are required to replicate the paper honestly.

## D003 — Extension: post-publication holdout (2017 → latest)
- **Chose:** Seal everything after the paper's sample and test it once, at the end.
- **Rejected:** Picking an extension after seeing replication results.
- **Why:** Choosing the extension before seeing data avoids a forking-paths problem, and
  "does it survive after publication" is a question a quant researcher actually cares about.
