# Football Copilot 2026/27
## Gameweek 1 Prediction Audit

### Initial snapshot

An initial pre-match Gameweek 1 prediction snapshot was generated on:

**18 August 2026 at 18:04 UK time**

During subsequent pre-match validation, it was identified that Ipswich was being treated using historical 2024/25 Premier League data because the club already existed in the historical dataset.

This was inconsistent with the treatment of the other 2026/27 promoted clubs, Coventry City and Hull City.

### Methodology correction

Before any Gameweek 1 matches were played, Ipswich was added to the promoted-team cold-start framework.

All three promoted clubs are therefore treated consistently using:

- 2025/26 Championship performance
- historical Championship-to-Premier-League translation factors
- 75% translated Championship estimate
- 25% historical Premier League baseline

The original snapshot has been retained as:

`2026_27_gw01_predictions_superseded_v1.csv`

and has not been altered.

A corrected pre-match snapshot will be generated before the opening Gameweek 1 fixture.