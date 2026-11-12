from nfl_injuries import (
    QB_ADJUSTMENT_SCALE,
    get_injury_adjustment,
    get_qb_availability,
    load_injury_report_data,
    load_qb_availability_context,
    load_qb_context,
    load_snap_history,
)
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import nflreadpy as nfl
from scipy.stats import norm
from nfl_weather import get_game_weather

# ==========================================
# NFL AUTOMATIC HANDICAPPING MODEL
# ==========================================

SEASON = 2026

HOME_ADVANTAGE = 1.5
EPA_TO_POINTS = 20.0
SPREAD_STD = 13.5
TOTAL_STD = 13.5

# Historical starting baseline.
# Replace with calculated historical averages
# after the backtesting module is added.
BASE_TOTAL = 44.0

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}


def download_data():
    print("\nDownloading NFL data...")

    pbp = nfl.load_pbp(SEASON)
    games = nfl.load_schedules([SEASON])

    return pbp.to_pandas(), games.to_pandas()


def prepare_plays(pbp, games):
    # Exclude games that have not finished.
    completed = games[
        (games["game_type"] == "REG")
        & games["home_score"].notna()
        & games["away_score"].notna()
    ]

    df = pbp[
        (pbp["season_type"] == "REG")
        & pbp["play_type"].isin(["pass", "run"])
        & pbp["epa"].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
        & pbp["game_id"].isin(completed["game_id"])
    ].copy()

    return df


def calculate_rankings(df):
    offense = df.groupby("posteam")["epa"].mean()
    defense = df.groupby("defteam")["epa"].mean()

    teams = pd.DataFrame({
        "off_epa": offense,
        "def_epa": defense
    }).dropna()

    teams["net_epa"] = (
        teams["off_epa"] - teams["def_epa"]
    )

    # Opponent strength based on season net EPA.
    # This is a first-pass SOS adjustment.
    opponent_strength = (
        df["defteam"]
        .map(teams["net_epa"])
    )

    sos = (
        df.assign(opponent_strength=opponent_strength)
        .groupby("posteam")["opponent_strength"]
        .mean()
    )

    teams["sos"] = sos.reindex(teams.index).fillna(0)

    # Center SOS around the league average.
    teams["sos"] -= teams["sos"].mean()

    # Last five games for each team.
    recent = df[
        ["game_id", "week", "posteam", "defteam", "epa"]
    ].copy()

    recent_off = (
        recent.groupby(["posteam", "game_id", "week"])["epa"]
        .mean()
        .reset_index()
        .sort_values(["posteam", "week", "game_id"])
    )

    recent_off = (
        recent_off.groupby("posteam")
        .tail(5)
        .groupby("posteam")["epa"]
        .mean()
    )

    teams["recent_off_epa"] = recent_off

    teams["recent_off_epa"] = (
        teams["recent_off_epa"]
        .fillna(teams["off_epa"])
    )

    # Experimental power rating.
    teams["raw_power"] = (
        teams["net_epa"] * EPA_TO_POINTS
        + teams["sos"] * 10
        + (
            teams["recent_off_epa"]
            - teams["off_epa"]
        ) * 5
    )

    # Average team has a rating of zero.
    teams["power"] = (
        teams["raw_power"]
        - teams["raw_power"].mean()
    )

    teams = teams.sort_values(
        "power",
        ascending=False
    )

    teams["rank"] = range(1, len(teams) + 1)

    return teams


def display_rankings(teams):
    print("\n" + "=" * 66)
    print("             NFL POWER RANKINGS")
    print("=" * 66)

    print(
        f"{'RK':<5}"
        f"{'TEAM':<7}"
        f"{'POWER':>9}"
        f"{'NET EPA':>11}"
        f"{'SOS':>10}"
        f"{'REC OFF':>11}"
    )

    print("-" * 66)

    for team, row in teams.iterrows():
        print(
            f"{int(row['rank']):<5}"
            f"{team:<7}"
            f"{row['power']:>+9.2f}"
            f"{row['net_epa']:>+11.3f}"
            f"{row['sos']:>+10.3f}"
            f"{row['recent_off_epa']:>+11.3f}"
        )


def get_upcoming_games(games, now=None):
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    schedule = games.copy()

    schedule["gameday"] = pd.to_datetime(
        schedule["gameday"],
        errors="coerce"
    )

    schedule["gametime"] = (
        schedule["gametime"].fillna("00:00")
    )

    schedule["kickoff"] = pd.to_datetime(
        schedule["gameday"].dt.strftime("%Y-%m-%d")
        + " "
        + schedule["gametime"],
        errors="coerce"
    ).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")

    # Select the next regular-season week with at least one future kickoff.
    future = schedule[
        (schedule["game_type"] == "REG")
        & schedule["home_score"].isna()
        & schedule["away_score"].isna()
        & schedule["kickoff"].notna()
        & (schedule["kickoff"] > now_utc)
    ].copy()

    if future.empty:
        return future

    next_week = future.sort_values("kickoff").iloc[0]["week"]

    return future[
        future["week"] == next_week
    ].sort_values("kickoff")


def prediction_week_mismatch(predictions, upcoming):
    if (
        predictions.empty
        or upcoming.empty
        or "week" not in predictions.columns
        or "week" not in upcoming.columns
    ):
        return False

    expected_weeks = pd.to_numeric(
        upcoming["week"], errors="coerce"
    ).dropna().astype(int).unique()
    predicted_weeks = pd.to_numeric(
        predictions["week"], errors="coerce"
    ).dropna().astype(int).unique()
    if len(expected_weeks) != 1 or len(predicted_weeks) == 0:
        return False
    return any(week != expected_weeks[0] for week in predicted_weeks)


def download_odds():
    if not ODDS_API_KEY:
        print("\nNo Odds API key configured.")
        print("Sportsbook comparisons will be unavailable.")
        return {}

    url = (
        "https://api.the-odds-api.com/v4/sports/"
        "americanfootball_nfl/odds/"
    )

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()
        events = response.json()

    except Exception as error:
        print("Unable to retrieve sportsbook odds:", error)
        return {}

    odds = {}

    for event in events:
        home = event["home_team"]
        away = event["away_team"]

        spreads = []
        totals = []

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                outcomes = market.get("outcomes", [])

                if market["key"] == "spreads":
                    for outcome in outcomes:
                        if outcome["name"] == home:
                            spreads.append(outcome["point"])

                elif market["key"] == "totals":
                    for outcome in outcomes:
                        if outcome["name"] == "Over":
                            totals.append(outcome["point"])

        odds[(away, home)] = {
            "spread": float(np.median(spreads))
            if spreads else None,
            "total": float(np.median(totals))
            if totals else None
        }

    return odds


def predict_game(
    home,
    away,
    teams,
    kickoff,
    week,
    injury_reports,
    snap_history,
    gsis_to_pfr,
    qb_context,
    qb_availability_context=None,
    qb_report_history=None,
    injury_report_weeks=None,
):
    home_power = teams.loc[home, "power"]
    away_power = teams.loc[away, "power"]

    home_injury_adjustment = get_injury_adjustment(
        home,
        week,
        injury_reports,
        SEASON,
        snap_history,
        gsis_to_pfr,
        qb_context,
        kickoff,
        QB_ADJUSTMENT_SCALE,
    )
    away_injury_adjustment = get_injury_adjustment(
        away,
        week,
        injury_reports,
        SEASON,
        snap_history,
        gsis_to_pfr,
        qb_context,
        kickoff,
        QB_ADJUSTMENT_SCALE,
    )
    report_history = qb_report_history or {}
    home_qb_availability = get_qb_availability(
        home,
        week,
        SEASON,
        kickoff,
        qb_availability_context,
        report_history,
        injury_report_weeks,
    )
    away_qb_availability = get_qb_availability(
        away,
        week,
        SEASON,
        kickoff,
        qb_availability_context,
        report_history,
        injury_report_weeks,
    )

    margin = (
        home_power
        - away_power
        + HOME_ADVANTAGE
    )

    margin += home_injury_adjustment - away_injury_adjustment

    # Initial total projection based on offensive
    # and defensive EPA deviations from league averages.
    league_off = teams["off_epa"].mean()
    league_def = teams["def_epa"].mean()

    home_off = teams.loc[home, "off_epa"] - league_off
    away_off = teams.loc[away, "off_epa"] - league_off

    home_def = teams.loc[home, "def_epa"] - league_def
    away_def = teams.loc[away, "def_epa"] - league_def

    total = (
        BASE_TOTAL
        + 20 * (
            home_off + away_off
            + home_def + away_def
        )
    )
    # Weather adjustments  to projected total 
    weather = get_game_weather(home, kickoff)
    
    if weather.get("status")== "Forecast available":
        total+= weather.get("adjustment", 0.0)
    
    home_score = (total + margin) / 2
    away_score = (total - margin) / 2

    home_win = norm.cdf(margin / SPREAD_STD)

    return {
        "home": home,
        "away": away,
        "home_rank": int(teams.loc[home, "rank"]),
        "away_rank": int(teams.loc[away, "rank"]),
        "home_power": home_power,
        "away_power": away_power,
        "home_injury_adjustment": home_injury_adjustment,
        "away_injury_adjustment": away_injury_adjustment,
        "home_qb_starter": home_qb_availability["starter_name"],
        "home_qb_starter_confidence": home_qb_availability[
            "starter_confidence"
        ],
        "home_qb_starter_status": home_qb_availability["starter_status"],
        "home_qb_replacement": home_qb_availability["replacement_name"],
        "home_qb_replacement_status": home_qb_availability[
            "replacement_status"
        ],
        "home_qb_uncertainty": home_qb_availability["uncertainty"],
        "home_qb_latest_roster_week": home_qb_availability["latest_roster_week"],
        "home_qb_latest_injury_report_week": home_qb_availability[
            "latest_injury_report_week"
        ],
        "home_qb_current_week_roster_coverage": home_qb_availability[
            "current_week_roster_coverage"
        ],
        "home_qb_current_week_injury_coverage": home_qb_availability[
            "current_week_injury_coverage"
        ],
        "home_qb_coverage_warning": home_qb_availability["coverage_warning"],
        "home_qb_prior_availability_flags": home_qb_availability[
            "prior_qb_availability_flags"
        ],
        "home_qb_team_assignment_warnings": home_qb_availability[
            "team_assignment_warnings"
        ],
        "away_qb_starter": away_qb_availability["starter_name"],
        "away_qb_starter_confidence": away_qb_availability[
            "starter_confidence"
        ],
        "away_qb_starter_status": away_qb_availability["starter_status"],
        "away_qb_replacement": away_qb_availability["replacement_name"],
        "away_qb_replacement_status": away_qb_availability[
            "replacement_status"
        ],
        "away_qb_uncertainty": away_qb_availability["uncertainty"],
        "away_qb_latest_roster_week": away_qb_availability["latest_roster_week"],
        "away_qb_latest_injury_report_week": away_qb_availability[
            "latest_injury_report_week"
        ],
        "away_qb_current_week_roster_coverage": away_qb_availability[
            "current_week_roster_coverage"
        ],
        "away_qb_current_week_injury_coverage": away_qb_availability[
            "current_week_injury_coverage"
        ],
        "away_qb_coverage_warning": away_qb_availability["coverage_warning"],
        "away_qb_prior_availability_flags": away_qb_availability[
            "prior_qb_availability_flags"
        ],
        "away_qb_team_assignment_warnings": away_qb_availability[
            "team_assignment_warnings"
        ],
        "margin": margin,
        "home_spread": -margin,
        "total": total,
        "home_score": home_score,
        "away_score": away_score,
        "home_win": home_win,
        "weather_status": weather.get("status"),
	"temperature": weather.get("temperature"),
	"wind": weather.get("wind"),
	"gusts": weather.get("gusts"),
	"rain_chance": weather.get("rain_chance"),
	"snowfall": weather.get("snowfall"),
	"weather_adjustment": weather.get ("adjustment", 0.0),
    }


def analyze_games(
    games,
    teams,
    odds,
    injury_reports,
    snap_history,
    gsis_to_pfr,
    qb_context,
    qb_availability_context=None,
    qb_report_history=None,
    injury_report_weeks=None,
):
    results = []

    print("\n" + "=" * 72)
    print("              AUTOMATIC NFL GAME ANALYSIS")
    print("=" * 72)

    for _, game in games.iterrows():
        home = game["home_team"]
        away = game["away_team"]

        if home not in teams.index or away not in teams.index:
            continue
        
        prediction = predict_game(
            home,
            away,
            teams,
            game["kickoff"],
            int(game["week"]),
            injury_reports,
            snap_history,
            gsis_to_pfr,
            qb_context,
            qb_availability_context,
            qb_report_history,
            injury_report_weeks,
        )

        home_name = TEAM_NAMES.get(home)
        away_name = TEAM_NAMES.get(away)

        market = odds.get(
            (away_name, home_name),
            {}
        )

        market_spread = market.get("spread")
        market_total = market.get("total")

        print(f"\n{away} at {home}")

        print(
            f"Rankings: {away} #{prediction['away_rank']}"
            f" | {home} #{prediction['home_rank']}"
        )

        print(
            f"Power: {away} {prediction['away_power']:+.2f}"
            f" | {home} {prediction['home_power']:+.2f}"
        )
        print(
            f"Injury adjustments: {away} "
            f"{prediction['away_injury_adjustment']:+.2f}"
            f" | {home} {prediction['home_injury_adjustment']:+.2f}"
        )

        print(
            f"Model home spread: "
            f"{prediction['home_spread']:+.1f}"
        )

        print(
            f"Projected total: "
            f"{prediction['total']:.1f}"
        )

        print(f"Weather: {prediction['weather_status']}")
        print(f"Temperature: {prediction['temperature']} F")
        print(f"Wind: {prediction['wind']} mph")
        print(f"Gusts: {prediction['gusts']} mph")
        print(f"Rain chance: {prediction['rain_chance']}%")
        print(f"Weather adjustment: {prediction['weather_adjustment']:+.1f} points")
	

        print(
            f"Projected score: "
            f"{away} {prediction['away_score']:.1f}, "
            f"{home} {prediction['home_score']:.1f}"
        )

        print(
            f"{home} win probability: "
            f"{prediction['home_win']:.1%}"
        )

        if market_spread is not None:
            spread_edge = (
                market_spread
                - prediction["home_spread"]
            )

            print(f"Market home spread: {market_spread:+.1f}")
            print(f"Home spread edge: {spread_edge:+.1f}")

            prediction["market_spread"] = market_spread
            prediction["spread_edge"] = spread_edge

        if market_total is not None:
            total_edge = (
                prediction["total"] - market_total
            )

            print(f"Market total: {market_total:.1f}")
            print(f"Total edge: {total_edge:+.1f}")

            prediction["market_total"] = market_total
            prediction["total_edge"] = total_edge

            over_probability = norm.cdf(
                total_edge / TOTAL_STD
            )

            print(
                f"Estimated over probability: "
                f"{over_probability:.1%}"
            )
        
        prediction["week"] = int(game["week"])
        results.append(prediction)

    return results


def main():
    print("\nNFL AUTOMATIC HANDICAPPING MODEL")
    print(f"Season: {SEASON}")

    try:
        pbp, games = download_data()
    except Exception as error:
        print("Data download failed:", error)
        return

    df = prepare_plays(pbp, games)

    if df.empty:
        print("No completed regular-season play data found.")
        return

    teams = calculate_rankings(df)

    display_rankings(teams)

    upcoming = get_upcoming_games(games)

    if upcoming.empty:
        print("\nNo upcoming regular-season games found.")
        return

    injury_reports, qb_report_history, injury_report_weeks = (
        load_injury_report_data(SEASON)
    )
    qb_availability_context = load_qb_availability_context(SEASON)




    # Identify teams without qualifying injury statuses.
    prediction_week = int(upcoming["week"].iloc[0])
    upcoming_teams = set(upcoming["home_team"]) | set(upcoming["away_team"])

    missing_injury_teams = sorted(
        team for team in upcoming_teams
        if (SEASON, prediction_week, team) not in injury_reports
    )

    if missing_injury_teams:
        print(
            "\nINJURY DATA WARNING: No qualifying injury statuses for "
            + ", ".join(missing_injury_teams)
        )
        print(
            "This does not confirm these teams are healthy. "
            "Injury adjustments may be incomplete."
        )

    snap_history, gsis_to_pfr = load_snap_history(SEASON)
    qb_context = (
        load_qb_context(SEASON, games)
        if QB_ADJUSTMENT_SCALE is not None
        else None
    )
    odds = download_odds()

    results = analyze_games(
        upcoming,
        teams,
        odds,
        injury_reports,
        snap_history,
        gsis_to_pfr,
        qb_context,
        qb_availability_context,
        qb_report_history,
        injury_report_weeks,
    )

    if results:
        output = pd.DataFrame(results)

        output.to_csv(
            "nfl_predictions.csv",
            index=False
        )

        print("\nPredictions saved to nfl_predictions.csv")

    print("\nMODEL LIMITATIONS:")
    print("- Injury adjustments use weekly report statuses when available.")
    if QB_ADJUSTMENT_SCALE is None:
        print("- QB replacement estimates use the fixed QB penalty until calibrated.")
    print("- Historical market calibration is not yet incorporated.")
    print("- SOS adjustment is preliminary.")
    print("- Total projections use an assumed baseline.")
    print("- Probabilities are not backtested.")
    print("- No betting recommendation is implied.")


if __name__ == "__main__":
    main()
