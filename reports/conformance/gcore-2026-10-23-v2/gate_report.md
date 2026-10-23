# Gate report: G-core — PASS

- run seed: `gcore-2026-10-23-v2`
- trace dir: `traces/gcore-v2`
- replay seconds: 1842.4

| check | value | requirement | ok |
|---|---|---|---|
| battles | 50000 | >= 50000 | yes |
| CBR | 1.0 | >= 0.995 | yes |
| MPR | 1.0 | >= 1.0 | yes |
| DPR | 1.0 | >= 1.0 | yes |
| error_rate | 0.0 | <= 0.0 | yes |
| coverage:ACCURACY | 590046 | >= 50 | yes |
| coverage:CRIT | 1437119 | >= 50 | yes |
| coverage:DAMAGE_ROLL | 1437119 | >= 50 | yes |
| coverage:MULTIHIT_COUNT | 23344 | >= 50 | yes |
| coverage:RANDOM_TARGET | 0 | >= 50 (crafted: 360 hits, tests/scenarios/core/test_struggle_random_target.py) | yes |
| coverage:SECONDARY | 616293 | >= 50 | yes |
| coverage:SPEED_TIE_SELECT | 19605 | >= 50 | yes |
| coverage:SPEED_TIE_SWITCHIN | 31476 | >= 50 | yes |
| coverage:SPEED_TIE_FIELD_EVENT | 0 | >= 50 (waived) | yes |

## Top buckets

- none

## Waivers

[{"gate": "G-core", "target": "SPEED_TIE_FIELD_EVENT", "adr": "ADR-102", "probe": "oracle/tools/probes/core_field_event_handlers.js", "reason": "no core-stage entity can create a fieldEvent handler", "approved": "2026-10-23"}, {"gate": "G-core", "target": "RANDOM_TARGET", "adr": "ADR-102", "covered_by": "tests/scenarios/core/test_struggle_random_target.py", "hits": 360, "reason": "non-degenerate random targets need a 0-PP Struggle with two living foes; crafted corpus 400/400 clean", "approved": "2026-10-23"}]

## Contract hashes (sha256)
verified against docs/decisions/contract-hashes.md: True

- `docs/coverage/waivers.json`: `124659a06bb87f7fdb6e23e9da3657b5d5b6d1ec9e8084252b044d6e6498e055`
- `docs/design/contract/drawkey.md`: `4b714d94a9f38b39b97ff0efae685d4da6dfaa86cfc84e3c9765e767c1167116`
- `harness/contract/canonicalisation.json`: `fd8d9786a52d701a1fee68815a08ec18d19f61ee07e9ed475d3faadb6a688302`
- `harness/contract/degenerate.json`: `25acbaa6712772fb81e264735620bed54eb2d7f92dd8512546922386ae313f23`
- `harness/contract/dontcare.json`: `dfc9d227be68409cf260408cef292a5fc66a0cc6411b0ed81b92754ee17becb4`
- `harness/contract/normalisations.json`: `d39868914e6f0bc3f6108db8aed360b6019d3bf4fa0140657d4b239c9be9d7c1`
- `harness/contract/projection_fields.json`: `a618b71dd0c6421618a7b0b66b5d768464e6b91a553f8f3eab22a9c085b2f3d9`
- `harness/gates.yaml`: `1d3c03f076f3aa49330c0feb2f13ec80bdb012742e8cc62e2f223e121f2a87a4`
