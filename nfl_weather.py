import requests
import pandas as pd

# Approximate stadium coordinates.
# "indoor" means weather does not affect play.
# Retractable roofs default to indoor until
# actual roof status is confirmed.

STADIUMS = {
    "ARI": (33.5276, -112.2626, "indoor"),
    "ATL": (33.7554, -84.4008, "indoor"),
    "BAL": (39.2780, -76.6227, "outdoor"),
    "BUF": (42.7740, -78.7870, "outdoor"),
    "CAR": (35.2258, -80.8528, "outdoor"),
    "CHI": (41.8623, -87.6167, "outdoor"),
    "CIN": (39.0954, -84.5160, "outdoor"),
    "CLE": (41.5061, -81.6995, "outdoor"),
    "DAL": (32.7480, -97.0928, "indoor"),
    "DEN": (39.7439, -105.0201, "outdoor"),
    "DET": (42.3400, -83.0456, "indoor"),
    "GB": (44.5013, -88.0622, "outdoor"),
    "HOU": (29.6847, -95.4107, "indoor"),
    "IND": (39.7601, -86.1639, "indoor"),
    "JAX": (30.3239, -81.6373, "outdoor"),
    "KC": (39.0489, -94.4839, "outdoor"),
    "LA": (33.9535, -118.3392, "covered"),
    "LAC": (33.9535, -118.3392, "covered"),
    "LV": (36.0909, -115.1833, "indoor"),
    "MIA": (25.9580, -80.2389, "outdoor"),
    "MIN": (44.9738, -93.2575, "indoor"),
    "NE": (42.0909, -71.2643, "outdoor"),
    "NO": (29.9511, -90.0812, "indoor"),
    "NYG": (40.8135, -74.0745, "outdoor"),
    "NYJ": (40.8135, -74.0745, "outdoor"),
    "PHI": (39.9008, -75.1675, "outdoor"),
    "PIT": (40.4468, -80.0158, "outdoor"),
    "SEA": (47.5952, -122.3316, "outdoor"),
    "SF": (37.4030, -121.9700, "outdoor"),
    "TB": (27.9759, -82.5033, "outdoor"),
    "TEN": (36.1665, -86.7713, "outdoor"),
    "WAS": (38.9078, -76.8645, "outdoor"),
}


def get_game_weather(home_team, kickoff):
    stadium = STADIUMS.get(home_team)

    if stadium is None:
        return {"status": "Unknown stadium", "adjustment": 0.0}

    lat, lon, stadium_type = stadium

    if stadium_type in ("indoor", "covered"):
        return {
            "status": f"{stadium_type.title()} stadium",
            "adjustment": 0.0
        }

    kickoff = pd.to_datetime(kickoff, utc=True)

    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": (
                    "temperature_2m,wind_speed_10m,"
                    "wind_gusts_10m,precipitation_probability,"
                    "precipitation,snowfall"
                ),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "timeformat": "unixtime",
                "forecast_days": 16
            },
            timeout=15
        )
        response.raise_for_status()

        hourly = response.json()["hourly"]
        times = pd.to_datetime(
            hourly["time"], unit="s", utc=True
        )

        differences = abs(times - kickoff)
        index = differences.argmin()

        if differences[index] > pd.Timedelta(minutes=90):
            return {
                "status": "Forecast unavailable for kickoff",
                "adjustment": 0.0
            }

        def value(name):
            result = hourly[name][index]
            return float(result) if result is not None else None

        temperature = value("temperature_2m")
        wind = value("wind_speed_10m")
        gusts = value("wind_gusts_10m")
        rain_chance = value("precipitation_probability")
        precipitation = value("precipitation")
        snowfall = value("snowfall")

        if any(x is None for x in (
            temperature, wind, gusts, rain_chance,
            precipitation, snowfall
        )):
            return {
                "status": "Incomplete weather forecast",
                "adjustment": 0.0
            }

        # Experimental totals adjustment.
        # These coefficients are NOT backtested.
        adjustment = 0.0

        if wind >= 20:
            adjustment -= 3.0
        elif wind >= 15:
            adjustment -= 1.5

        if gusts >= 30:
            adjustment -= 1.0

        if precipitation >= 0.10:
            adjustment -= 1.0

        if snowfall >= 0.5:
            adjustment -= 1.5

        if temperature <= 20:
            adjustment -= 1.0

        return {
            "status": "Forecast available",
            "temperature": temperature,
            "wind": wind,
            "gusts": gusts,
            "rain_chance": rain_chance,
            "precipitation": precipitation,
            "snowfall": snowfall,
            "adjustment": adjustment
        }

    except (requests.RequestException, ValueError, KeyError) as error:
        return {
            "status": f"Weather unavailable: {error}",
            "adjustment": 0.0
        }
