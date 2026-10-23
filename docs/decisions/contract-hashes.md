# Contract hashes (ADR-029)

sha256 of every protected contract file. Gate reports recompute these; a mismatch marks the verdict
UNVERIFIED CONTRACT. Written by the agent on the human's explicit instruction (2026-10-23: "write it");
update only together with a human-approved ADR that changes a protected file, and name that ADR below.

| file | sha256 |
|---|---|
| `docs/coverage/waivers.json` | `124659a06bb87f7fdb6e23e9da3657b5d5b6d1ec9e8084252b044d6e6498e055` |
| `docs/design/contract/drawkey.md` | `96e87a45e7f7c31af6404a9b3e580d7a6cc90d976aa9864732ce79d0c7375c46` |
| `harness/contract/canonicalisation.json` | `fd8d9786a52d701a1fee68815a08ec18d19f61ee07e9ed475d3faadb6a688302` |
| `harness/contract/degenerate.json` | `25acbaa6712772fb81e264735620bed54eb2d7f92dd8512546922386ae313f23` |
| `harness/contract/dontcare.json` | `b2b312294a10dc974ef16f7e8ef94ae654a572b619f8807c63b87e34105eb7ef` |
| `harness/contract/normalisations.json` | `2195a6653637a48677202fdc8bbe25b2588d3311a38cd8912ee7b62188c827d2` |
| `harness/contract/projection_fields.json` | `3af18a5b0f134eb70b1f9ebb3d6f54e17197fa191fa1bf8650b50e8afd5a1861` |
| `harness/gates.yaml` | `1d3c03f076f3aa49330c0feb2f13ec80bdb012742e8cc62e2f223e121f2a87a4` |

## Changes

- 2026-10-23: initial (contract_version 1; includes the ADR-102 waivers).
