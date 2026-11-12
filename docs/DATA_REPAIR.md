# Data repair: first pass

## Current findings

The checked-in injury source has 37,667 records and ends on **2023-04-16**.
The legacy game logs contain **25,887 appearances from 168 players**, spanning
2022-10-18 through 2025-04-13. The 2023-24 and 2024-25 seasons therefore cannot
receive valid negative labels from this source. The original cohort was selected
through injury records, so it must be replaced with full-league appearances.

The rebuilt study audit identifies **567 lower-body IL placement reports**:
**565 resolve to player identities**, and two remain malformed. Of the resolved
reports, **155 belong to players missing from the legacy cohort**. Counts are
report records, not independent injuries, and use the broader body-region rules
listed in `src/data/repair.py` (including foot, hip/groin, thigh and leg).

Unresolved source records (the source's `Unnamed: 0` values) are:

| Record | Date | Team | Value in the player-name field |
|---|---|---|---|
| 36723 | 2023-01-06 | Lakers | strained left quadriceps (DTD) |
| 37126 | 2023-02-27 | Magic | left knee injury (DTD) |

Do not infer their identities from a plausible injury or roster. Correct them
only after locating supporting source records. For now, unresolved target reports
block negative labels for overlapping future horizons.

## Run the audit and rebuild

From the repository root, with pandas and numpy installed:

```bash
python -m scripts.repair_data --tracking data/raw/tracking_partial.csv
python -m unittest discover -s tests -v
```

No network calls are made during repair. Results go to `data/processed/v2/`:

| File | Purpose |
|---|---|
| `audit.json` | Input hashes, coverage, matching counts, and readiness |
| `coverage_by_season.csv` | Season-level exposure and unknown-label counts |
| `report_identity_audit.csv` | All study reports, candidates, and match decisions |
| `target_reports.csv` | Lower-body placement reports, including unresolved records |
| `candidate_player_games.csv` | Rebuilt features and nullable future labels |
| `training_player_games.csv` | Eligible labeled rows, or header only when blocked |
| `feature_schema.json` | Explicit feature allowlist and target-only columns |

The default run is **audit only**: all supervised targets remain unknown because
coverage has not been independently verified. `observed_future_report_days` records
known reports for inspection, but is target information and must never be a model
feature. `feature_schema.json` identifies the permitted features.

## Prediction contract

Prediction occurs **after a completed game**. The target is a lower-body IL
placement report on a subsequent calendar date within seven days:

`GAME_DATE < report_date <= GAME_DATE + 7 days`

Same-day reports are excluded because source timestamps are date-only. This is a
forecast of reports, not verified injury onset, new episodes, or non-contact
mechanism. Recovery/surgery mentions are flagged but not silently removed.
Reports on the same player-date count once for the binary outcome. Repeated
placements on later dates are retained; episode reconstruction remains future work.

Name matching preserves suffixes, accents are normalized, collisions remain
unresolved, and manually reviewed aliases require evidence and can be date/team
scoped. Features reset at player-season boundaries. Day windows use calendar
dates, include the completed game, and cannot borrow other players' observations.
Zero-minute/short appearances and missing tracking remain in the cohort. Box-score
minutes are used when tracking minutes are absent. Missing statistics remain
missing, with observed-count features; fit imputation on training data later.

## Replace the selected-player cohort

```bash
python -m scripts.collect_full_cohort --seasons 2022-23
python -m scripts.repair_data --games data/raw/full_cohort/gamelogs.csv --cohort-manifest data/raw/full_cohort/cohort_manifest.json
```

The collector requests the NBA Stats `leaguegamelog` endpoint in player and team
modes. It does not select players from injury records. It checks the full
1,230-game regular-season schedule, reconciles team-games, rejects duplicates,
and checks at least five players per team-game. This checks structural completeness,
not every individual box-score value. It retains raw responses and request metadata
and emits a SHA-256-bound manifest only after successful collection.

The default schedule check is intended for completed 82-game seasons. Do not use
it unchanged for shortened or current seasons. A failed refresh invalidates the
manifest and cannot silently certify a previous download.

Live collection was attempted during this repair, but this execution environment
could not resolve `stats.nba.com`. No full-league collection was completed here.
For now, use 2022-23 as the first repair scope, since the current injury source
does not extend into the following two seasons. The old tracking file covers the
selected cohort only; omit it for a full-league minutes-based baseline, or supply
a full tracking extraction later. Never inner-join tracking to define the cohort.

## Verify injury reporting coverage

`config/injury_coverage.json` intentionally contains no verified windows. A minimum
or maximum observed record date is not evidence of complete reporting. Obtain
source documentation or audit a source with known observation coverage before
enabling negative labels. Confirm that it covers all players in the study.

After verification, record an exact input hash and evidenced windows:

```json
{
  "source_sha256": "SHA256_OF_THE_INJURY_CSV",
  "windows": [
    {
      "start": "YYYY-MM-DD",
      "end": "YYYY-MM-DD",
      "verified": true,
      "evidence": "Source URL and description of the completeness verification"
    }
  ]
}
```

Do not mark a window verified just to make training run. The repair command
requires the configured hash to match the actual injury file. Only horizons
fully contained in verified reporting windows can be labeled. A verified full
cohort and at least one eligible labeled row are also required for training
export. Unknown horizons are excluded and all readiness decisions are audited.

## Remaining work before retraining

1. Complete the full-league game-log download.
2. Establish verified injury-source coverage or replace the source.
3. Resolve or explicitly retain the two malformed source records.
4. Review target reports and decide how to define independent injury episodes.
5. Wire all models to the same v2 target, feature schema and chronological splits.

The old processed dataset and report are retained as historical artifacts. Their
metrics must not be presented as results of this repaired pipeline.
