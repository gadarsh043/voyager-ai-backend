# Travel AI Backend

Python backend for generating trip itineraries. Uses **Ollama** (local) to produce 3 itinerary options and returns JSON matching the frontend spec.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional: install and run Ollama locally ([ollama.ai](https://ollama.ai)), then pull a model, e.g. `ollama pull llama3.2`. If Ollama is not running, the API returns a valid fallback response.

## Run

```bash
uvicorn main:app --reload
```

- API: `http://localhost:8000`
- OpenAPI: `http://localhost:8000/docs`

## Endpoint

- **POST /itinerary/generate**  
  - **Body:** Trip parameters (all optional). Examples: `origin`, `destination`, `start_date`, `end_date`, `budget`, `preferences` (list of interests). Empty `{}` uses defaults (e.g. Tokyo, 3–4 days, moderate budget).  
  - **Response:** `{ "options": [ { "id", "label", "daily_plan", "total_estimated_cost" }, ... ] }` with `daily_plan` containing `flight_from_source`, `flight_to_origin`, `hotel_stay`, `days` (see spec in repo).

CORS is enabled for `http://localhost:5173` and `http://localhost:3000` so the React app can call the API.
