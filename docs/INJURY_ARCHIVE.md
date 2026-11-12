# Historical game-status archive audit

## Inputs and provenance

The full 2022-23 regular-season cohort comes directly from NBA Stats'
`leaguegamelog` endpoint in player and team modes. No injury-based player selection
is used. Source URLs, retrieval times, response hashes and normalized CSV hashes
are recorded in `data/raw/cohort_2022_23/cohort_manifest.json`.

The injury-status CSV was downloaded from Vaughn Hajra's historical archive:
https://statsurge.substack.com/p/downloadable-nba-injury-datasets

The author's 2022-23 regular-season download is a published Google Sheets CSV
(sheet `2111042600`). Its exact URL and byte hash are retained in
`data/raw/injury_reports_2022_23.metadata.json`. Credit for compiling this archive
belongs to Vaughn Hajra. The raw download is retained without modifications.
The NBA's official PDF reports are a primary source for spot checks, including:

- https://ak-static.cms.nba.com/referee/injury/Injury-Report_2023-01-06_06PM.pdf
- https://ak-static.cms.nba.com/referee/injury/Injury-Report_2023-02-27_06PM.pdf

These spot checks do not certify completeness across the season. In particular,
the February 27 report lists Jonathan Isaac's right hamstring condition, which
does not justify guessing an identity for the malformed Kaggle left-knee record.

The CSV's `DATE` is a **scheduled game date**. It is not an injury onset or
report-publication timestamp. The archive cannot be passed to the IL-transaction
label builder by renaming columns.

## Reproduce

```bash
python -m scripts.audit_injury_archive
python -m unittest discover -s tests -v
```

The normalized cohort, team schedule, injury CSV and identity references are
included, so this audit requires no network or Parquet dependency. For a fresh
download, run the Node command in DATA_REPAIR.md. Copy its
`responses/injury_reports_2022_23.csv` and companion metadata to `data/raw/` before
auditing the refreshed source. Never overwrite the saved metadata without also
updating the source bytes it identifies.

Generated outputs are in `data/processed/archive_audit/`; compact audit snapshots
are committed in `results/data_audit/`.

## Findings

| Check | Result |
|---|---:|
| Official regular-season games | 1,230 |
| Official participating players | 539 |
| Official appearances | 25,894 |
| Archived status rows | 10,692 |
| Status rows mapped to a scheduled game and team | 10,692 |
| Status rows mapped to a player ID | 10,692 |
| Team-games with at least one archive row | 2,325 / 2,460 |
| Team-games with no archive row | 135 |
| Out statuses contradicted by an official appearance | 45 |

The 135 team-games may include healthy teams or gaps in the archive. They are
not proof of either. An observed status row also does not prove that all affected
players were reported. All 45 participation conflicts are flagged, including
G League listings and medical reasons. The archive contains no timestamp that
would support choosing a later status to resolve the conflict.

Names preserve suffixes. Explicit, evidenced aliases are scoped to the 2022-23
team and date range. This handles historical names such as Brandon Boston Jr.,
Jeff Dowtin, Kenyon Martin Jr. and Jeenathan Williams without generic suffix
removal. Player identities use official season names before static-catalog fallbacks.

## Exploratory next-game absence outcome

`next_game_outcomes_exploratory.csv` contains one opportunity after each recorded
appearance. It asks whether the player has a documented lower-body absence in
the current team's next scheduled game, provided that game is within seven days.
It is an alternative research target, not a replacement silently applied to v2.

| Label | Required evidence |
|---|---|
| 1 | Out status for a lower-body medical reason; no official appearance in that game |
| 0 | Official appearance, or Out status documenting a different cause of absence |
| Unknown | Missing/uncertain absence reason, conflicting records, team change, or no next game within the horizon |

The observed outcomes comprise **376 positives, 22,179 negatives and 3,339 unknowns**.
Unknown rows remain nullable and carry an explicit reason. Eight of the flagged
Out/appearance conflicts correspond to these prediction opportunities; the other
conflicts occur on games not reached by an eligible previous-appearance anchor.

The label measures documented injury-related absence, not whether a player was
uninjured when they played. A player can play while injured. Recovery, surgery
and injury-management listings can be positives; they are not new injuries.

All next-game IDs/dates, outcome statuses, source row pointers and labels are
evaluation metadata. They must not become model features. Current-game workload
features can be generated separately with `engineer_features` and joined by
PLAYER_ID/GAME_ID.

Missing outcomes may be informative: for example, bench players may miss games
without an injury report. Dropping all unknowns can change the evaluation
population. This table is suitable for reviewing an explicit research design;
it is not evidence for unbiased full-league injury probabilities. No models have
been retrained on it, and the original seven-day IL task remains blocked.
