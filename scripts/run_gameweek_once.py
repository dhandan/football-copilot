"""Football Copilot: one manual, immutable gameweek freeze.

Run from repo root: python scripts/run_gameweek_once.py --gameweek 7 --dry-run
Official: python scripts/run_gameweek_once.py --gameweek 7

Dry run: FPL fixture/player data from latest archived H07 snapshot; no odds API calls.
Official: one fresh official FPL snapshot and one live UK h2h Odds API call.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from prediction import fixture_predictor as fp

load_dotenv(ROOT / '.env')
SEASON = '2026/27'
SEASON_FILE = '2026_27'
MODEL_VERSION = 'Model2_v1.0'
SHADOW_VERSION = 'Model2_OpeningMarket_40_60_v1.0'
ALIASES = {
    'Man Utd': 'Man United', 'Manchester United': 'Man United',
    'Manchester City': 'Man City', 'Nottingham Forest': "Nott'm Forest",
    'Spurs': 'Tottenham', 'Tottenham Hotspur': 'Tottenham',
    'Newcastle United': 'Newcastle', 'Brighton and Hove Albion': 'Brighton',
    'Brighton & Hove Albion': 'Brighton', 'West Ham United': 'West Ham',
    'Leeds United': 'Leeds', 'Ipswich Town': 'Ipswich',
    'Wolverhampton Wanderers': 'Wolves', 'AFC Bournemouth': 'Bournemouth',
    'Hull City AFC': 'Hull City', 'Coventry City FC': 'Coventry City',
}

def team_name(v):
    return ALIASES.get(str(v), str(v))

def fetch_json(url, params=None):
    try:
        r = requests.get(
            url,
            params=params,
            timeout=40,
            headers={'User-Agent': 'FootballCopilot/1.0'}
        )
        r.raise_for_status()
        return r.json(), r.headers
    except requests.exceptions.RequestException as exc:
        # Never expose the exception or request URL:
        # query parameters may contain the Odds API key.
        host = 'Odds API' if 'the-odds-api.com' in url else 'FPL API'
        error_type = type(exc).__name__
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        detail = f'HTTP {status}' if status is not None else error_type
        raise RuntimeError(
            f'{host} request failed ({detail}). Check VPN, connectivity and credentials.'
        ) from None

def fpl_data(dry):
    if dry:
        root = ROOT / 'data/live/players/h07_snapshots'
        folders = sorted((p for p in root.iterdir() if p.is_dir() and
                         (p/'bootstrap.json').exists() and (p/'fixtures.json').exists()), reverse=True)
        if not folders:
            raise RuntimeError('No saved FPL bootstrap/fixture snapshot for dry run')
        folder = folders[0]
        return json.loads((folder/'bootstrap.json').read_text(encoding='utf-8')), json.loads((folder/'fixtures.json').read_text(encoding='utf-8')), str(folder)
    bootstrap, _ = fetch_json('https://fantasy.premierleague.com/api/bootstrap-static/')
    fixtures, _ = fetch_json('https://fantasy.premierleague.com/api/fixtures/')
    return bootstrap, fixtures, 'FPL live API'

def get_fixtures(bootstrap, all_fixtures, gw):
    teams = {int(t['id']): team_name(t['name']) for t in bootstrap['teams']}
    selected = [x for x in all_fixtures if x.get('event') == gw]
    if len(selected) != 10:
        raise RuntimeError(f'GW{gw}: expected 10 fixtures, received {len(selected)}')
    rows = []
    for x in selected:
        if x.get('finished'):
            raise RuntimeError(f'Fixture {x["id"]} already finished; cannot freeze')
        kickoff = pd.Timestamp(x['kickoff_time'])
        if kickoff.tzinfo is None:
            raise RuntimeError('FPL kickoff must include UTC timezone')
        kickoff = kickoff.tz_convert('UTC')
        uk = kickoff.tz_convert('Europe/London')
        rows.append(dict(FixtureId=int(x['id']), HomeTeam=teams[int(x['team_h'])],
                         AwayTeam=teams[int(x['team_a'])], KickoffUTC=kickoff,
                         FixtureDate=uk.strftime('%Y-%m-%d'), FixtureTime=uk.strftime('%H:%M')))
    df = pd.DataFrame(rows).sort_values('KickoffUTC').reset_index(drop=True)
    if df['FixtureId'].duplicated().any() or df[['HomeTeam','AwayTeam']].isna().any().any():
        raise RuntimeError('Duplicate/invalid fixtures')
    return df

def get_history(capture):
    base = pd.read_csv(ROOT/'data/processed/matches_clean.csv')
    base['Date'] = pd.to_datetime(base['Date'], errors='raise')
    base = base[base['Date'].dt.date < capture.date()].copy()
    src = []
    for p in sorted((ROOT/'data/live/results').glob('2026_27_gw*_results.csv')):
        d = pd.read_csv(p)
        required = {'Season','FixtureId','FixtureDate','HomeTeam','AwayTeam','Status','ActualHomeGoals','ActualAwayGoals'}
        if not required.issubset(d.columns):
            raise RuntimeError(f'{p}: missing result fields')
        d = d[d['Status'] == 'FINISHED'].copy()
        d['Date'] = pd.to_datetime(d['FixtureDate'], errors='raise')
        d = d[d['Date'].dt.date < capture.date()]
        if d.empty: continue
        if d[['ActualHomeGoals','ActualAwayGoals']].isna().any().any():
            raise RuntimeError(f'{p}: missing completed score')
        part = pd.DataFrame({
            'Season': d['Season'].astype(str), 'Date': d['Date'],
            'HomeTeam': d['HomeTeam'].map(team_name), 'AwayTeam': d['AwayTeam'].map(team_name),
            'FTHG': pd.to_numeric(d['ActualHomeGoals']), 'FTAG': pd.to_numeric(d['ActualAwayGoals'])
        })
        part['FTR'] = np.where(part['FTHG'] > part['FTAG'], 'H', np.where(part['FTHG'] < part['FTAG'], 'A', 'D'))
        src.append(part)
    current = pd.concat(src, ignore_index=True) if src else pd.DataFrame(columns=base.columns)
    if len(current) and not (current['Season'] == SEASON).all():
        raise RuntimeError('Unexpected season in live results')
    for c in base.columns:
        if c not in current.columns: current[c] = np.nan
    merged = pd.concat([base, current[base.columns]], ignore_index=True)
    if merged.duplicated(['Season','Date','HomeTeam','AwayTeam']).any():
        raise RuntimeError('Duplicate historical/current matches')
    return merged.sort_values('Date').reset_index(drop=True), len(current)

def model_forecasts(fixtures, history, captured, gw, model_sha):
    original = fp.matches
    rows = []
    try:
        fp.matches = history
        for f in fixtures.itertuples(index=False):
            pr = fp.predict_fixture(f.HomeTeam, f.AwayTeam)
            h, d, a = [float(pr[k]) for k in ('home_win_probability','draw_probability','away_win_probability')]
            if not (abs(h + d + a - 100) <= .25 and min(h,d,a) >= 0):
                raise RuntimeError(f'Bad Model2 probabilities: {f.HomeTeam} vs {f.AwayTeam}')
            lead = (f.KickoffUTC-captured).total_seconds()/3600
            if lead <= 0:
                raise RuntimeError('Cannot freeze after kickoff')
            rows.append(dict(PredictionTimestamp=captured.isoformat(), Season=SEASON, Gameweek=gw,
                FixtureIdSource='FPL', FixtureId=f.FixtureId, FixtureDate=f.FixtureDate, FixtureTime=f.FixtureTime,
                HomeTeam=f.HomeTeam, AwayTeam=f.AwayTeam, ModelVersion=MODEL_VERSION,
                PredictionStatus='Predicted', PredictedResult=[f.HomeTeam,'Draw',f.AwayTeam][np.argmax([h,d,a])],
                MostLikelyScore=(pr.get('most_likely_scores') or [{}])[0].get('score'),
                HomeWinProbability=h, DrawProbability=d, AwayWinProbability=a,
                ExpectedHomeGoals=float(pr['expected_home_goals']), ExpectedAwayGoals=float(pr['expected_away_goals']),
                ColdStartUsed=bool(pr.get('feature_metadata',{}).get('cold_start_used',False)),
                ColdStartTeams='|'.join(pr.get('feature_metadata',{}).get('cold_start_teams',[])),
                ColdStartMethod=pr.get('feature_metadata',{}).get('cold_start_method'),
                FeatureSeason=pr.get('feature_season'), ActualHomeGoals=None, ActualAwayGoals=None,
                ActualResult=None, Evaluated=False, KickoffUTC=f.KickoffUTC.isoformat(),
                LeadHours=round(lead,2), T24Eligible=bool(lead >= 24), ModelSHA256=model_sha))
    finally:
        fp.matches = original
    result = pd.DataFrame(rows)
    if len(result) != 10 or not result['FeatureSeason'].eq(SEASON).all():
        raise RuntimeError('Model2 did not generate ten current-season predictions')
    return result

def market_forecasts(fixtures, captured, gw, api_key):
    date_from = fixtures['KickoffUTC'].min().floor('D').strftime('%Y-%m-%d')
    date_to = fixtures['KickoffUTC'].max().ceil('D').strftime('%Y-%m-%d')
    data, headers = fetch_json('https://api.the-odds-api.com/v4/sports/soccer_epl/odds/', {
        'apiKey':api_key,'regions':'uk','markets':'h2h','oddsFormat':'decimal','dateFormat':'iso',
        'commenceTimeFrom':f'{date_from}T00:00:00Z','commenceTimeTo':f'{date_to}T23:59:59Z'
    })
    by_fixture = {(r.HomeTeam,r.AwayTeam):r for r in fixtures.itertuples(index=False)}
    output=[]; seen=set()
    for event in data:
        key=(team_name(event['home_team']),team_name(event['away_team']))
        if key not in by_fixture: continue
        if key in seen: raise RuntimeError(f'Duplicate odds fixture: {key}')
        fixture = by_fixture[key]
        odds_kickoff = pd.Timestamp(event['commence_time']).tz_convert('UTC')
        if abs((fixture.KickoffUTC-odds_kickoff).total_seconds())>600:
            raise RuntimeError(f'Kickoff mismatch for {key}: FPL={fixture.KickoffUTC} odds={odds_kickoff}')
        raw=[]; bookmakers=[]
        for bk in event.get('bookmakers',[]):
            mk=next((x for x in bk.get('markets',[]) if x.get('key')=='h2h'),None)
            if mk is None:continue
            prices={o['name']:float(o['price']) for o in mk.get('outcomes',[])}
            if not all(n in prices and prices[n]>1 for n in (event['home_team'],'Draw',event['away_team'])): continue
            raw.append([prices[event['home_team']],prices['Draw'],prices[event['away_team']]])
            bookmakers.append(bk['title'])
        if not raw: raise RuntimeError(f'No complete bookmaker odds for {key}')
        avg=np.mean(raw,axis=0); inv=1/avg; fair=inv/inv.sum()
        output.append(dict(Season=SEASON,Gameweek=gw,OddsEventId=event['id'],FixtureDate=fixture.FixtureDate,
          FixtureTime=fixture.FixtureTime,HomeTeam=key[0],AwayTeam=key[1],SnapshotTimestamp=captured.isoformat(),
          MarketSource='The Odds API live UK h2h (Friday snapshot, not necessarily opening odds)',Region='uk',Market='h2h',
          BookmakerCount=len(bookmakers),Bookmakers=' | '.join(bookmakers),AvgH=avg[0],AvgD=avg[1],AvgA=avg[2],
          Overround=inv.sum(),MarketHomeProbability=fair[0],MarketDrawProbability=fair[1],MarketAwayProbability=fair[2],
          KickoffUTC=fixture.KickoffUTC.isoformat()))
        seen.add(key)
    if len(output)!=10: raise RuntimeError(f'Only {len(output)} of 10 matches have matched odds')
    print('Odds API requests remaining:',headers.get('x-requests-remaining','not reported'))
    return pd.DataFrame(output)

def blend_forecasts(model, market, captured, gw):
    keys=['FixtureDate','FixtureTime','HomeTeam','AwayTeam']
    merged=model.merge(market,on=keys,how='outer',validate='one_to_one',indicator=True,suffixes=('','_Market'))
    if len(merged)!=10 or not merged['_merge'].eq('both').all():
        raise RuntimeError('Model2 and odds fixtures do not match 10/10')
    merged=merged.drop(columns='_merge')
    m=merged[['HomeWinProbability','DrawProbability','AwayWinProbability']].to_numpy(dtype=float)/100
    m=m/m.sum(axis=1,keepdims=True)
    k=merged[['MarketHomeProbability','MarketDrawProbability','MarketAwayProbability']].to_numpy(dtype=float)
    z=.4*m+.6*k
    if not np.allclose(z.sum(axis=1),1,atol=1e-6): raise RuntimeError('Invalid blended probability')
    for i,name in enumerate(['ShadowHomeProbability','ShadowDrawProbability','ShadowAwayProbability']): merged[name]=z[:,i]
    merged['ShadowPredictedResult']=np.array(['H','D','A'])[z.argmax(axis=1)]
    merged.insert(0,'ChallengerVersion',SHADOW_VERSION)
    merged.insert(1,'ModelWeight',.4)
    merged.insert(2,'MarketWeight',.6)
    merged.insert(3,'SnapshotType','OFFICIAL')
    return merged

def save_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as out: out.write(content)

def csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gameweek',type=int,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    captured=pd.Timestamp.now(tz='UTC')
    prefix=f'{SEASON_FILE}_gw{args.gameweek:02d}'
    outputs={
      'model':ROOT/f'data/live/predictions/{prefix}_predictions.csv',
      'market':ROOT/f'data/live/market/{prefix}_opening_market_official_{captured.strftime("%Y%m%d_%H%M%S")}.csv',
      'shadow':ROOT/f'data/live/challenger_predictions/{prefix}_market_shadow_predictions.csv',
    }
    if not args.dry_run:
        existing=[str(p) for name,p in outputs.items() if p.exists()]
        # Also prevent second official market version if a previous run stopped partway.
        existing += [str(p) for p in (ROOT/'data/live/market').glob(f'{prefix}_opening_market_official_*.csv')]
        if existing:raise RuntimeError('Existing official snapshots: '+', '.join(sorted(set(existing))))
    bootstrap,fixtures_json,fpl_source=fpl_data(args.dry_run)
    fpl_retrieved=pd.Timestamp.now(tz='UTC') if not args.dry_run else None
    fixtures=get_fixtures(bootstrap,fixtures_json,args.gameweek)
    if not args.dry_run and captured.date().isoformat()!= '2026-10-16' and args.gameweek==7:
        raise RuntimeError('GW7 official run is reserved for Friday 16 October 2026 UTC; use --dry-run today')
    if (fixtures['KickoffUTC'] <= captured).any(): raise RuntimeError('Some fixture kickoffs have passed')
    hist,current_count=get_history(captured)
    model_path=ROOT/'models/production_model_v2.pkl'
    model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    model=model_forecasts(fixtures,hist,captured,args.gameweek,model_hash)
    print(f'GW{args.gameweek} {"DRY RUN" if args.dry_run else "OFFICIAL"}: 10/10 Model2, '+
          f'{current_count} current-season results; T-24 eligible {int(model.T24Eligible.sum())}/10; FPL: {fpl_source}')
    print(model[['HomeTeam','AwayTeam','LeadHours','T24Eligible','HomeWinProbability','DrawProbability','AwayWinProbability']].to_string(index=False))
    if args.dry_run:
        print('DRY RUN PASS: no official files, no odds API call, no snapshots saved')
        return
    api_key=os.getenv('ODDS_API_KEY')
    if not api_key:raise RuntimeError('Missing ODDS_API_KEY in .env; no files written')
    market=market_forecasts(fixtures,captured,args.gameweek,api_key)
    odds_retrieved=pd.Timestamp.now(tz='UTC')
    market['SnapshotTimestamp']=odds_retrieved.isoformat()
    shadow=blend_forecasts(model,market,captured,args.gameweek)
    # GW8 onward: unchanged 40/60 blend is the provisional champion.
    # Model 2 remains the frozen benchmark.
    model['ForecastRole'] = 'Benchmark'
    shadow['ForecastRole'] = 'ProvisionalChampion'

    stamp=captured.strftime('%Y%m%dT%H%M%S%fZ')
    fpl_folder=ROOT/'data/live/players/h07_snapshots'/stamp
    if fpl_folder.exists(): raise FileExistsError(str(fpl_folder))
    manifest={'CapturedUTC':captured.isoformat(),'Gameweek':args.gameweek,'Policy':'single manual weekly Friday snapshot',
              'T24EligibleCount':int(model.T24Eligible.sum()),'ModelSHA256':model_hash,
              'FPLRetrievedUTC':fpl_retrieved.isoformat(),
              'OddsRetrievedUTC':odds_retrieved.isoformat(),
              'FixtureIdSource':'Fantasy Premier League; existing evaluator uses different IDs',
              'HistoricMatchesUsed':len(hist),'CurrentSeasonMatchesUsed':current_count,
              'H07':'observational only; no injury/replacement goal adjustments',
              'Market':'UK h2h pre-match odds at Friday capture, not guaranteed opening'}
    manifest['PrimaryForecastVersion'] = SHADOW_VERSION
    manifest['BenchmarkForecastVersion'] = MODEL_VERSION
    manifest['PromotionStatus'] = 'Provisional'
    # All remote reads and integrity checks precede writes. Keep official outputs write-once.
    for name,df in [('model',model),('market',market),('shadow',shadow)]:
        if len(df)!=10:raise RuntimeError('Incomplete '+name)
    fpl_folder.mkdir(parents=True,exist_ok=False)
    save_once(fpl_folder/'bootstrap.json',json.dumps(bootstrap).encode('utf-8'))
    save_once(fpl_folder/'fixtures.json',json.dumps(fixtures_json).encode('utf-8'))
    save_once(fpl_folder/'players.csv',csv_bytes(pd.DataFrame(bootstrap['elements'])))
    save_once(fpl_folder/'teams.csv',csv_bytes(pd.DataFrame(bootstrap['teams'])))
    save_once(fpl_folder/'manifest.json',json.dumps(manifest,indent=2).encode('utf-8'))
    for name,df in [('model',model),('market',market),('shadow',shadow)]:
        save_once(outputs[name],csv_bytes(df))
        print('SAVED:',outputs[name])
    print('SAVED H07 player snapshot:',fpl_folder)
    print('OFFICIAL COMPLETE: 10/10 fixtures; model and 40/60 weights unchanged')

if __name__=='__main__': main()
