# Player identity reference

`nba_players.csv` contains PLAYER_ID and PLAYER_NAME extracted from the `players`
literal in nba_api's static catalog, without executing the source file.

Source: https://github.com/swar/nba_api/blob/ec190700dab27498b3f75ecac8f932e3a3ee1606/src/nba_api/stats/library/data.py

The upstream catalog is MIT licensed; its notice is retained in NBA_API_LICENSE.txt.
The catalog is a spelling/identity reference, not a season roster or cohort.
Names are not assumed unique. Generational suffixes are preserved.

`player_aliases.csv` contains explicit date/team-scoped identity corrections, with
official NBA profile evidence. Johnny Davis is scoped to the 2022-23 Wizards to
avoid confusing him with the retired player with the same name. Expanding the
study dates requires reviewing these scopes. Malformed source names are left
unresolved rather than inferred from the injury description.
