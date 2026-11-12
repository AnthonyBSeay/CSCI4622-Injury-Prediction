/** Download raw sources with Node's built-in HTTP client (Node 18+).
 * No packages or browser session are required. Python validates cached responses
 * independently before accepting the cohort.
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { createHash } from 'node:crypto';

const season = process.argv[2] ?? '2022-23';
const output = process.argv[3] ?? 'data/raw/cohort_2022_23/responses';
if (!/^\d{4}-\d{2}$/.test(season)) throw new Error('Expected season YYYY-YY');
await mkdir(output, { recursive: true });
const jobs = ['P', 'T'].map(mode => {
  const parameters = new URLSearchParams({Counter:'0', DateFrom:'', DateTo:'', Direction:'DESC', LeagueID:'00', PlayerOrTeam:mode, Season:season, SeasonType:'Regular Season', Sorter:'DATE'});
  return [`${season}_${mode}.json`, `https://stats.nba.com/stats/leaguegamelog?${parameters}`];
});
jobs.push(['injury_reports_2022_23.csv', 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTI7Y9RhrP7BhS59Nsofqnb7QqOlqXotOAKQV8Tjyv0WHF2p2yoTsF6LIVBk9vOS2oWBs5Yt4Jxp2no/pub?gid=2111042600&single=true&output=csv']);
for (const [file, url] of jobs) {
  const response = await fetch(url, {signal:AbortSignal.timeout(60000), headers:{'User-Agent':'Mozilla/5.0', Referer:'https://www.nba.com/'}});
  if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  await writeFile(join(output, file), bytes);
  const metadata = {url, sha256:createHash('sha256').update(bytes).digest('hex'), bytes:bytes.length, retrieved_at:new Date().toISOString()};
  await writeFile(join(output, file.replace(/\.[^.]+$/, '') + '.metadata.json'), JSON.stringify(metadata, null, 2) + '\n');
  console.log(`${file}: ${bytes.length} bytes`);
}
