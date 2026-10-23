# Gate report: G-core — PASS

- run seed: `gcore-2026-10-21-a`
- trace dir: `traces/gcore`
- replay seconds: 590.9

| check | value | requirement | ok |
|---|---|---|---|
| battles | 50000 | >= 50000 | yes |
| CBR | 1.0 | >= 0.995 | yes |
| MPR | 1.0 | >= 1.0 | yes |
| DPR | 1.0 | >= 1.0 | yes |
| error_rate | 0.0 | <= 0.0 | yes |
| coverage:ACCURACY | 589508 | >= 50 | yes |
| coverage:CRIT | 1436330 | >= 50 | yes |
| coverage:DAMAGE_ROLL | 1436330 | >= 50 | yes |
| coverage:MULTIHIT_COUNT | 23496 | >= 50 | yes |
| coverage:RANDOM_TARGET | 0 | >= 50 (crafted: 360 hits, tests/scenarios/core/test_struggle_random_target.py) | yes |
| coverage:SECONDARY | 614146 | >= 50 | yes |
| coverage:SPEED_TIE_SELECT | 20025 | >= 50 | yes |
| coverage:SPEED_TIE_SWITCHIN | 31849 | >= 50 | yes |
| coverage:SPEED_TIE_FIELD_EVENT | 0 | >= 50 (waived) | yes |

## Top buckets

- none

## Waivers

[{"gate": "G-core", "target": "SPEED_TIE_FIELD_EVENT", "adr": "ADR-102", "probe": "oracle/tools/probes/core_field_event_handlers.js", "reason": "no core-stage entity can create a fieldEvent handler", "approved": "2026-10-23"}, {"gate": "G-core", "target": "RANDOM_TARGET", "adr": "ADR-102", "covered_by": "tests/scenarios/core/test_struggle_random_target.py", "hits": 360, "reason": "non-degenerate random targets need a 0-PP Struggle with two living foes; crafted corpus 400/400 clean", "approved": "2026-10-23"}]

## Contract hashes (sha256)
verified against docs/decisions/contract-hashes.md: True

- `docs/coverage/waivers.json`: `124659a06bb87f7fdb6e23e9da3657b5d5b6d1ec9e8084252b044d6e6498e055`
- `docs/design/contract/drawkey.md`: `96e87a45e7f7c31af6404a9b3e580d7a6cc90d976aa9864732ce79d0c7375c46`
- `harness/contract/canonicalisation.json`: `fd8d9786a52d701a1fee68815a08ec18d19f61ee07e9ed475d3faadb6a688302`
- `harness/contract/degenerate.json`: `25acbaa6712772fb81e264735620bed54eb2d7f92dd8512546922386ae313f23`
- `harness/contract/dontcare.json`: `b2b312294a10dc974ef16f7e8ef94ae654a572b619f8807c63b87e34105eb7ef`
- `harness/contract/normalisations.json`: `2195a6653637a48677202fdc8bbe25b2588d3311a38cd8912ee7b62188c827d2`
- `harness/contract/projection_fields.json`: `3af18a5b0f134eb70b1f9ebb3d6f54e17197fa191fa1bf8650b50e8afd5a1861`
- `harness/gates.yaml`: `1d3c03f076f3aa49330c0feb2f13ec80bdb012742e8cc62e2f223e121f2a87a4`
