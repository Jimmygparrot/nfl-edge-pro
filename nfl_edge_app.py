import streamlit as st
from pathlib import Path
import pandas as pd
import nflreadpy as nfl
from nfl_model import get_upcoming_games, prediction_week_mismatch


@st.cache_data(ttl=300)
def load_next_eligible_slate(season):
    try:
        schedule = nfl.load_schedules([season]).to_pandas()
    except Exception:
        return None
    return get_upcoming_games(schedule)


@st.cache_data(ttl=3600)
def get_latest_injury_week(season=2026):
    try:
        injuries = nfl.load_injuries(season)

        if injuries.is_empty():
            return None

        usable = injuries.filter(
            nfl_status_filter(injuries)
        )

        if usable.is_empty():
            return None

        return int(usable["week"].max())

    except Exception:
        return None


def nfl_status_filter(injuries):
    import polars as pl

    return (
        pl.col("report_status")
        .str.to_uppercase()
        .is_in(["OUT", "DOUBTFUL", "QUESTIONABLE"])
    )




@st.cache_data(ttl=3600)
def load_injury_coverage(season):
    import polars as pl

    injuries = nfl.load_injuries(season)

    if injuries.is_empty():
        return pd.DataFrame()

    return (
        injuries
        .filter(pl.col("season_type") == "REG")
        .select([
            "week",
            "team",
            "full_name",
            "position",
            "practice_primary_injury",
            "practice_status",
            "report_status",
        ])
        .to_pandas()
    )


st.set_page_config(
    page_title="NFL Edge Pro",
    page_icon="🏈",
    layout="wide",
)

st.title("🏈 NFL Edge Pro")
st.caption("NFL Prediction & Analytics Dashboard")

rankings_file = Path(__file__).parent / "nfl_predictions.csv"
predictions_file = rankings_file

st.divider()

st.subheader("Prediction Center")


if st.button("🏈 Run All Predictions", type="primary"):
    import subprocess
    import sys
    from pathlib import Path

    with st.spinner("Running NFL prediction model..."):
        result = subprocess.run(
            [sys.executable, "nfl_model.py"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=False,
        )
    
    if result.returncode == 0:
        get_latest_injury_week.clear()
        load_injury_coverage.clear()
        load_next_eligible_slate.clear()

        st.success("NFL predictions updated successfully!")
        st.rerun()

    
    else:
        st.error("NFL prediction model failed.")
        st.code(result.stderr or result.stdout)

if rankings_file.exists():
    saved_predictions = pd.read_csv(rankings_file)
    next_slate = load_next_eligible_slate(2026)
    if next_slate is None:
        st.warning(
            "Unable to download the NFL schedule. Prediction freshness cannot "
            "be verified; the saved CSV is displayed without confirmation "
            "that it matches the current slate."
        )
    elif prediction_week_mismatch(saved_predictions, next_slate):
        saved_weeks = sorted(
            pd.to_numeric(saved_predictions["week"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        scheduled_week = int(next_slate["week"].iloc[0])
        st.warning(
            f"Saved predictions contain Week {', '.join(map(str, saved_weeks))}, "
            f"but the next eligible scheduled slate is Week {scheduled_week}. "
            "The displayed predictions may be stale; run predictions to refresh."
        )


st.divider()

col1, col2, col3 = st.columns([1.2, 1, 1.5])



with col1:
    st.subheader("🏆 Power Rankings")

    if rankings_file.exists():
        rankings = pd.read_csv(rankings_file)

        home_teams = rankings[
            ["home", "home_rank", "home_power"]
        ].copy()
        home_teams.columns = ["Team", "Rank", "Power"]

        away_teams = rankings[
            ["away", "away_rank", "away_power"]
        ].copy()
        away_teams.columns = ["Team", "Rank", "Power"]

        team_rankings = pd.concat(
            [home_teams, away_teams],
            ignore_index=True
        )

        team_rankings = (
            team_rankings
            .drop_duplicates(subset=["Team"])
            .sort_values("Rank")
        )

        st.dataframe(
            team_rankings,
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("Run predictions to see rankings.")


with col2:
    st.subheader("🏥 Injury Adjustments")
    
    
    latest_injury_week = get_latest_injury_week(2026)
    
    prediction_week = None
    if rankings_file.exists():
        prediction_data = pd.read_csv(rankings_file)
        if "week" in prediction_data.columns and not prediction_data.empty:
            prediction_week = int(prediction_data["week"].max())

    


 
    if latest_injury_week is None:
        st.warning("Unable to verify injury-report availability.")
    elif prediction_week is not None and latest_injury_week < prediction_week:
        st.warning(
            
f"Week {prediction_week} has no usable injury game-status reports yet. "
f"Latest week with usable statuses: Week {latest_injury_week}. "
"Injury adjustments may be incomplete. Zero does not necessarily mean no injuries."

        )




    if rankings_file.exists():
        injury_data = pd.read_csv(rankings_file)

        home_injuries = injury_data[
            ["home", "home_injury_adjustment"]
        ].copy()
        home_injuries.columns = ["Team", "Adjustment"]

        away_injuries = injury_data[
            ["away", "away_injury_adjustment"]
        ].copy()
        away_injuries.columns = ["Team", "Adjustment"]

        injury_summary = pd.concat(
            [home_injuries, away_injuries],
            ignore_index=True
        )

        st.dataframe(
            injury_summary,
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("Run predictions to see injury adjustments.")



with col3:
    st.subheader("🌦️ Weather & SOS")

    if rankings_file.exists():
        weather_data = pd.read_csv(rankings_file)

        weather_columns = [
            "away",
            "home",
            "weather_status",
            "temperature",
            "wind",
            "rain_chance",
            "weather_adjustment",
        ]

        available_columns = [
            col for col in weather_columns
            if col in weather_data.columns
        ]

        st.dataframe(
            weather_data[available_columns],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Run predictions to see weather data.")



st.divider()
st.subheader("🏈 NFL Game Predictions")

if predictions_file.exists():
    predictions = pd.read_csv(predictions_file)
    st.dataframe(predictions, use_container_width=True, hide_index=True)
else:
    st.warning("No predictions found yet. Run the NFL model first.")


st.divider()
st.subheader("🏈 Game Analysis Center")

if predictions_file.exists():
    games = pd.read_csv(predictions_file)

    matchups = (
        games["away"] + " @ " + games["home"]
    )

    selected_game = st.selectbox(
        "Select an NFL matchup",
        matchups
    )

    game = games.loc[matchups == selected_game].iloc[0]

    st.subheader(selected_game)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric(
            "Predicted Score",
            f"{game['away_score']:.1f} - {game['home_score']:.1f}"
        )

    with col_b:
        st.metric(
            "Home Spread",
            f"{game['home_spread']:+.1f}"
        )

    with col_c:
        st.metric(
            "Projected Total",
            f"{game['total']:.1f}"
        )

# Injury Report Coverage
st.divider()
st.subheader("Injury Report Coverage")


if prediction_week is not None:
    coverage = load_injury_coverage(2026)

    if coverage.empty:
        st.warning("No injury reports available.")
    else:
        coverage = coverage[
            coverage["week"] == prediction_week
        ]

        if coverage.empty:
            st.warning(
                f"No Week {prediction_week} injury reports available."
            )
        
        else:
            # Classify injury-report data availability
            usable_statuses = [
                "OUT",
                "DOUBTFUL",
                "QUESTIONABLE",
            ]

            coverage["Report Availability"] = (
                coverage["report_status"]
                .fillna("")
                .str.upper()
                .isin(usable_statuses)
                .map({
                    True: "Game status available",
                    False: "Preliminary / no usable status",
                })
            )

            def classify_practice(status):
                if pd.isna(status):
                    return "Unknown"

                status = str(status).lower()

                if "did not participate" in status:
                    return "High - Missed practice"
                elif "limited" in status:
                    return "Medium - Limited practice"
                elif "full" in status:
                    return "Low - Full practice"

                return "Unknown"

            coverage["Practice Concern"] = (
                coverage["practice_status"]
                .apply(classify_practice)
            )



            
            show_all = st.checkbox(
                "Show all injury reports",
                value=False,
            )

            if show_all:
                displayed_coverage = coverage
            else:
                displayed_coverage = coverage[
                    coverage["Practice Concern"].str.startswith(
                        ("High", "Medium")
                    )
                ]

            if displayed_coverage.empty:
                st.info(
                    "No High or Medium practice concerns "
                    "found in the available reports."
                )
            else:
                st.dataframe(
                    displayed_coverage,
                    hide_index=True,
                    use_container_width=True,
                )



            # Identify teams missing from injury reports
            predicted_teams = set(
                prediction_data["home"]
            ) | set(prediction_data["away"])

            reported_teams = set(coverage["team"])

            missing_teams = sorted(
                predicted_teams - reported_teams
            )

            st.caption(
                f"Teams with injury records: {len(predicted_teams & reported_teams)} "
                f"of {len(predicted_teams)}"
            )

            if missing_teams:
                st.warning(
                    "No injury records available for: "
                    + ", ".join(missing_teams)
                    + ". Missing data does not mean no injuries."
                )

# Team Injury Coverage Summary
if prediction_week is not None and rankings_file.exists():
    st.subheader("Team Injury Coverage Summary")

    predicted_teams = sorted(
        set(prediction_data["home"]) |
        set(prediction_data["away"])
    )

    rows = []

    for team in predicted_teams:
        if coverage.empty:
            team_reports = pd.DataFrame()
        else:
            team_reports = coverage[
                coverage["team"] == team
            ]

        total_records = len(team_reports)

        if total_records == 0:
            usable_count = 0
            status = "No report"
        else:
            usable_count = int(
                team_reports["report_status"]
                .fillna("")
                .str.upper()
                .isin(["OUT", "DOUBTFUL", "QUESTIONABLE"])
                .sum()
            )

            status = (
                "Game statuses available"
                if usable_count > 0
                else "Preliminary"
            )

        rows.append({
            "Team": team,
            "Injury Records": total_records,
            "Usable Game Statuses": usable_count,
            "Coverage Status": status,
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Coverage indicates data availability, not report completeness. "
        "No report does not mean no injuries."
    )

st.divider()
st.subheader("Quarterback Availability & Data Quality")

if predictions_file.exists():
    qb_predictions = pd.read_csv(predictions_file)
    qb_rows = []
    for side in ("away", "home"):
        required_columns = [
            f"{side}_qb_starter",
            f"{side}_qb_starter_confidence",
            f"{side}_qb_starter_status",
            f"{side}_qb_replacement",
            f"{side}_qb_replacement_status",
            f"{side}_qb_uncertainty",
            f"{side}_qb_latest_roster_week",
            f"{side}_qb_latest_injury_report_week",
            f"{side}_qb_current_week_roster_coverage",
            f"{side}_qb_current_week_injury_coverage",
            f"{side}_qb_coverage_warning",
            f"{side}_qb_prior_availability_flags",
            f"{side}_qb_team_assignment_warnings",
        ]
        if not all(column in qb_predictions.columns for column in required_columns):
            continue
        for _, row in qb_predictions.iterrows():
            qb_rows.append({
                "Team": row[side],
                "Projected QB1": row[f"{side}_qb_starter"],
                "Starter confidence": row[f"{side}_qb_starter_confidence"],
                "QB1 availability": row[f"{side}_qb_starter_status"],
                "Projected replacement": row[f"{side}_qb_replacement"],
                "Replacement status": row[f"{side}_qb_replacement_status"],
                "Missing / uncertain information": row[f"{side}_qb_uncertainty"],
                "Latest roster week": row[f"{side}_qb_latest_roster_week"],
                "Latest injury-feed week": row[
                    f"{side}_qb_latest_injury_report_week"
                ],
                "Current-week roster data": row[
                    f"{side}_qb_current_week_roster_coverage"
                ],
                "Current-week injury-feed data": row[
                    f"{side}_qb_current_week_injury_coverage"
                ],
                "QB data coverage note": row[f"{side}_qb_coverage_warning"],
                "Prior reserve / OUT records for lower-ranked QBs": row[
                    f"{side}_qb_prior_availability_flags"
                ],
                "QB team assignment validation": row[
                    f"{side}_qb_team_assignment_warnings"
                ],
            })

    if qb_rows:
        qb_data = pd.DataFrame(qb_rows)
        st.dataframe(qb_data, hide_index=True, use_container_width=True)
        st.caption(
            "Weekly roster status is a week-level snapshot without a "
            "transaction timestamp. Generic reserve status does not by itself "
            "confirm an Injured Reserve designation. Injury-feed week coverage "
            "means team records are present, not that the report is complete. "
            "These notes do not change predictions."
        )
        diagnostics = qb_data[
            (
                qb_data["QB data coverage note"].fillna("").str.strip() != ""
            )
            | (
                qb_data["QB team assignment validation"]
                .fillna("")
                .str.strip()
                != ""
            )
        ]
        if not diagnostics.empty:
            st.info(
                "Review QB team-assignment validation, data coverage, and "
                "prior reserve/OUT records above. These diagnostics are "
                "informational only and do not alter predictions."
            )
    else:
        st.warning(
            "Quarterback availability diagnostics are missing from the prediction file."
        )
else:
    st.info("Run predictions to see quarterback availability diagnostics.")
