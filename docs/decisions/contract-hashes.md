# Contract hashes (ADR-029)

sha256 of every protected contract file. Gate reports recompute these; a mismatch marks the verdict
UNVERIFIED CONTRACT. Written by the agent on the human's explicit instruction (2026-10-23: "write it");
update only together with a human-approved ADR that changes a protected file, and name that ADR below.

| file | sha256 |
|---|---|
| `docs/coverage/waivers.json` | `124659a06bb87f7fdb6e23e9da3657b5d5b6d1ec9e8084252b044d6e6498e055` |
| `docs/design/contract/drawkey.md` | `4b714d94a9f38b39b97ff0efae685d4da6dfaa86cfc84e3c9765e767c1167116` |
| `harness/contract/canonicalisation.json` | `fd8d9786a52d701a1fee68815a08ec18d19f61ee07e9ed475d3faadb6a688302` |
| `harness/contract/degenerate.json` | `25acbaa6712772fb81e264735620bed54eb2d7f92dd8512546922386ae313f23` |
| `harness/contract/dontcare.json` | `dfc9d227be68409cf260408cef292a5fc66a0cc6411b0ed81b92754ee17becb4` |
| `harness/contract/normalisations.json` | `d39868914e6f0bc3f6108db8aed360b6019d3bf4fa0140657d4b239c9be9d7c1` |
| `harness/contract/projection_fields.json` | `a618b71dd0c6421618a7b0b66b5d768464e6b91a553f8f3eab22a9c085b2f3d9` |
| `harness/gates.yaml` | `1d3c03f076f3aa49330c0feb2f13ec80bdb012742e8cc62e2f223e121f2a87a4` |

## Changes

- 2026-10-23: initial (contract_version 1; includes the ADR-102 waivers).
- 2026-10-23: contract_version 2 (ADR-101 option b: RUNSWITCH_INSERT_TIE; ADR-103: ext fields compared,
  switch SELECT ties keep fainted actors). Files: projection_fields.json, normalisations.json, dontcare.json, drawkey.md.
