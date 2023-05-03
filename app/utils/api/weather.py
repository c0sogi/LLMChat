import httpx
import orjson
from typing import Any, Literal
from fastapi import HTTPException


async def fetch_weather_data(
    lat: float,
    lon: float,
    api_key: str,
    source: Literal["openweathermap", "weatherbit", "climacell"],
) -> Any:
    base_url = {
        "openweathermap": "https://api.openweathermap.org/data/2.5/weather",
        "weatherbit": "https://api.weatherbit.io/v2.0/current",
        "climacell": "https://api.climacell.co/v3/weather/realtime",
    }[source]
    query_params = {
        "openweathermap": f"lat={lat}&lon={lon}&appid={api_key}",
        "weatherbit": f"lat={lat}&lon={lon}&key={api_key}",
        "climacell": f"lat={lat}&lon={lon}&unit_system=metric&apikey={api_key}",
    }[source]

    async with httpx.AsyncClient() as client:
        response = await client.get(base_url + "?" + query_params)
        if response.status_code == 200:
            weather_data = orjson.loads(response.content)
            print("weather_data:", weather_data)
            return weather_data
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error fetching data from {source}",
            )


def get_temperature(
    weather_data: dict,
    source: Literal["openweathermap", "weatherbit", "climacell"],
):
    if source == "openweathermap":
        temp = weather_data["main"]["temp"] - 273.15  # Convert from Kelvin to Celsius
    elif source == "weatherbit":
        temp = weather_data["data"][0]["temp"]
    elif source == "climacell":
        temp = weather_data["temp"]["value"]
    else:
        temp = None

    return temp
