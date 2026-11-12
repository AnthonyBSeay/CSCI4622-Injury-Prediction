# Official 2022-23 regular-season cohort

Source: https://stats.nba.com/stats/leaguegamelog

`gamelogs.csv` retains 25,894 official player appearances from 539 players,
without filtering through injury records. `team_games.csv` is the matching
2,460-row team schedule (1,230 games, 30 teams). Zero-minute appearances in the
NBA logs are retained; integer minutes can round a short appearance to zero.

`cohort_manifest.json` records source request URLs, raw JSON SHA-256 hashes,
retrieval times, normalized CSV hashes and validation results. The raw responses
can be regenerated with the Node downloader; response caches are ignored by Git.

Reproduce with the commands in `docs/DATA_REPAIR.md`. CSV source bytes are protected
from Git line-ending conversion so their hashes remain stable across platforms.
