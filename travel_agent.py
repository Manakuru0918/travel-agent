
import os
import requests
from dotenv import load_dotenv
from langchain.tools import StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv(override=True)

# Verify API keys
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment variables")
if not os.environ.get("WEATHER_API_KEY"):
    raise ValueError("WEATHER_API_KEY not found in environment variables")
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = "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"

# Currency Converter Tool
class CurrencyConversionInput(BaseModel):
    amount: float = Field(..., description="The amount of money to convert")
    from_currency: str = Field(..., description="The currency to convert from (e.g., USD)")
    to_currency: str = Field(..., description="The currency to convert to (e.g., EUR)")

def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if to_currency not in data["rates"]:
            return f"Error: {to_currency} not supported."
        rate = data["rates"][to_currency]
        converted_amount = round(amount * rate, 2)
        return f"{amount} {from_currency} = {converted_amount} {to_currency}"
    except requests.RequestException as e:
        return f"Currency conversion failed: {e}"

currency_conversion_tool = StructuredTool.from_function(
    name="currency_converter",
    description="Convert an amount from one currency to another",
    func=convert_currency,
    args_schema=CurrencyConversionInput
)

# Weather Forecast Tool
class WeatherInput(BaseModel):
    location: str = Field(..., description="City to get weather for")

def get_weather(location: str) -> str:
    api_key = os.environ.get("WEATHER_API_KEY")
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": api_key, "units": "metric"}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return f"The weather in {data['name']} is {data['weather'][0]['description']} with a temperature of {data['main']['temp']}°C."
    except requests.RequestException as e:
        return f"Weather fetch failed: {e}"

weather_tool = StructuredTool.from_function(
    name="weather_forecast",
    description="Get current weather for a location",
    func=get_weather,
    args_schema=WeatherInput
)

# Translation Tool
class TranslationInput(BaseModel):
    text: str = Field(..., description="The text to translate")
    target_language: str = Field(..., description="The target language code (e.g., 'fr' for French)")

def translate_text(text: str, target_language: str) -> str:
    base_url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": "en",
        "tl": target_language,
        "dt": "t",
        "q": text
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        translation = response.json()[0][0][0]
        return translation
    except requests.RequestException as e:
        return f"Translation failed: {e}"

translation_tool = StructuredTool.from_function(
    name="language_translator",
    description="Translate text from English to a specified target language",
    func=translate_text,
    args_schema=TranslationInput
)

# Booking Storage Tool
class BookingInput(BaseModel):
    booking_type: str = Field(..., description="Type of booking (e.g., hotel, attraction)")
    details: dict = Field(..., description="Details of the booking, including price in USD")

def add_booking(booking_type: str, details: dict) -> str:
    booking_manager.add_booking(booking_type, details)
    return f"Added {booking_type} booking: {details}"

booking_tool = StructuredTool.from_function(
    name="add_booking",
    description="Store a booking (e.g., hotel, attraction) with details",
    func=add_booking,
    args_schema=BookingInput
)

# Tavily Search Tool
tavily_search = TavilySearchResults(max_results=2)

# Booking Storage
class BookingManager:
    def __init__(self):
        self.bookings = []

    def add_booking(self, booking_type: str, details: dict):
        self.bookings.append({"type": booking_type, **details})

    def get_bookings(self) -> list:
        return self.bookings

    def calculate_total_cost(self) -> float:
        return sum(booking.get("price", 0) for booking in self.bookings)

booking_manager = BookingManager()

# LangChain Agent Setup
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a smart travel agent helping users with travel-related queries. For weather queries, use the weather tool to provide the current weather. For trip planning, create a customized itinerary using the provided tools. Here's how to proceed for trip planning:

📌 Tools:
- Use the weather tool to get current weather in the destination city.
- Use the currency conversion tool to convert from USD to the local currency.
- Use the translation tool to translate at least one activity description to the local language.
- Use the search tool to find budget-friendly hotels and local attractions.
- Use the add_booking tool to store confirmed bookings for hotels and attractions.

📝 If planning a trip and any of the following details are missing, politely ask for them:
- Destination city or country
- Travel dates or duration (e.g., 5 days)
- Budget in USD
- Interests or activity preferences (optional but helpful)

🗺 Trip Plan Must Include:
- A day-by-day itinerary with recommended activities, estimated costs (in USD and local currency), and bookings (hotel, attractions).
- Total estimated cost and confirmation that it stays within the user's specified budget.
- Current weather in the destination.
- At least one activity description translated into the local language.
- Clear structured output format.

📦 After planning, use the add_booking tool to store details of confirmed bookings:
- Example (hotel): {{'name': 'Hotel Sakura', 'price': 550, 'nights': 5}}
- Example (attraction): {{'name': 'Tokyo Skytree', 'price': 20}}

🎯 Goal: For weather queries, provide a concise weather report. For trip planning, create a realistic, fun, and budget-conscious trip that matches user preferences.
"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
tools = [currency_conversion_tool, weather_tool, translation_tool, tavily_search, booking_tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Execute Query
if __name__ == "__main__":
    input_query = "What is the weather in khaderpeta in anantapur?"
    response = agent_executor.invoke({"input": input_query})
    print("\n=== Response ===")
    print(response["output"])
    
    # Display stored bookings (if any)
    print("\n=== Stored Bookings ===")
    for booking in booking_manager.get_bookings():
        print(booking)
    print(f"Total Cost: ${booking_manager.calculate_total_cost():.2f}")

