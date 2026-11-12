import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch

df = pd.read_csv("nfl_predictions.csv")

output = "NFL_Handicapping_Report.pdf"

doc = SimpleDocTemplate(
    output,
    pagesize=landscape(letter),
    rightMargin=20,
    leftMargin=20,
    topMargin=25,
    bottomMargin=20
)

styles = getSampleStyleSheet()
elements = []

elements.append(
    Paragraph("NFL HANDICAPPING REPORT", styles["Title"])
)

elements.append(
    Paragraph(
        "EPA-Based Predictions | Weather Adjustments | Projected Scores",
        styles["Normal"]
    )
)

elements.append(Spacer(1, 12))

headers = [
    "Matchup",
    "Winner",
    "Power",
    "Spread",
    "Total",
    "Score",
    "Home Win",
    "Temp",
    "Wind",
    "Gusts",
    "Rain",
    "Wx Adj"
]

data = [headers]

def fmt(value, decimals=1):
    if pd.isna(value):
        return "-"
    return f"{value:.{decimals}f}"

for _, row in df.iterrows():

    matchup = f"{row['away']} @ {row['home']}"

    if row["home_score"] > row["away_score"]:
        winner = row["home"]
    elif row["away_score"] > row["home_score"]:
        winner = row["away"]
    else:
        winner = "TIE"

    power = (
        f"{fmt(row['away_power'])} / "
        f"{fmt(row['home_power'])}"
    )

    spread = fmt(row["home_spread"])

    total = fmt(row["total"])

    score = (
        f"{fmt(row['away_score'])} - "
        f"{fmt(row['home_score'])}"
    )

    win = row["home_win"]
    if pd.notna(win):
        win = f"{win * 100:.1f}%"
    else:
        win = "-"

    data.append([
        matchup,
        winner,
	power,
        spread,
        total,
        score,
        win,
        fmt(row["temperature"]),
        fmt(row["wind"]),
        fmt(row["gusts"]),
        fmt(row["rain_chance"], 0),
        fmt(row["weather_adjustment"])
    ])

table = Table(
    data,
    repeatRows=1,
    colWidths=[
        72, 45, 60, 42, 40, 62,
        52, 38, 38, 38, 38, 42
    ]    

)

table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, 0), 8),
    ("FONTSIZE", (0, 1), (-1, -1), 7),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
     [colors.white, colors.HexColor("#EDF3F8")]),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
]))

elements.append(table)

elements.append(Spacer(1, 12))

elements.append(
    Paragraph(
        "Note: Model projections are experimental. "
        "Weather adjustments and probabilities have not been backtested.",
        styles["Normal"]
    )
)

doc.build(elements)

print(f"PDF created: {output}")
