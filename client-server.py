# weather_server.py — a minimal MCP server exposing one tool
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("weather-server")

@mcp.tool()
def get_weather(city: str) -> str:
    """Gets the current weather for a given city."""
    fake_data = {
        "bangalore": "28°C, partly cloudy",
        "mumbai": "31°C, humid",
        "delhi": "36°C, clear skies",
    }
    key = city.strip().lower()
    return fake_data.get(key, f"No weather data available for {city}")

if __name__ == "__main__":
    mcp.run()