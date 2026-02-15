# Travel AI Backend

Python backend for generating trip itineraries. Uses **Ollama** (local) to produce 3 itinerary options and returns JSON matching the frontend spec.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ollama (local AI) – required for real itineraries

The app uses **Ollama** to generate itineraries. If Ollama is not running or the model is missing, the API returns a fallback response (generic places).

### 1. Install Ollama

- **macOS / Linux:** [ollama.com](https://ollama.ai) → download and install.
- **Windows:** Same from [ollama.com](https://ollama.ai).

### 2. Pull a model (choose one)

| Model | Command | Size | Best for |
|-------|--------|------|----------|
| **Llama 3.2 (3B)** *(default)* | `ollama pull llama3.2` | ~2 GB | Fast, low RAM; good for itineraries. |
| **Llama 3.1 (8B)** | `ollama pull llama3.1:8b` | ~4.7 GB | Better quality and variety; recommended if you have 8GB+ RAM. |
| **Llama 3.2 (1B)** | `ollama pull llama3.2:1b` | ~1.3 GB | Slowest machines only. |

```bash
# Recommended for best itinerary quality (needs ~8GB RAM):
ollama pull llama3.1:8b

# Default in this app (lighter, still good):
ollama pull llama3.2
```

### 3. Run Ollama (keep it running)

Ollama runs as a local server. After install it’s often already running. If not:

```bash
# Start the Ollama service (runs in background; port 11434)
ollama serve
# Or on many setups, just run any model and the server starts:
ollama run llama3.2
```

### 4. Use a different model (optional)

By default the app uses **llama3.2**. To use Llama 3.1 8B (or any other model):

```bash
export OLLAMA_MODEL=llama3.1:8b
uvicorn main:app --reload
```

## Run the backend

```bash
uvicorn main:app --reload
# Optional: expose with ngrok
# ngrok http 8000
```

- API: `http://localhost:8000`
- OpenAPI: `http://localhost:8000/docs`

## Endpoint

- **POST /itinerary/generate**  
  - **Body:** Trip parameters (all optional). Examples: `origin`, `destination`, `start_date`, `end_date`, `budget`, `preferences` (list of interests). Empty `{}` uses defaults (e.g. Tokyo, 3–4 days, moderate budget).  
  - **Response:** `{ "options": [ { "id", "label", "daily_plan", "total_estimated_cost" }, ... ] }` with `daily_plan` containing `flight_from_source`, `flight_to_origin`, `hotel_stay`, `days` (see spec in repo).

CORS is enabled for `http://localhost:5173` and `http://localhost:3000` so the React app can call the API.

---

## Building the best itinerary planner (what to do)

1. **Use a capable model** – Set `OLLAMA_MODEL=llama3.1:8b` (or a larger model if you have RAM) for richer, more varied itineraries.
2. **Keep Ollama running** – Ensure `ollama serve` or `ollama run <model>` is running so the app never falls back to generic plans.
3. **Send clear preferences** – The more specific `origin`, `destination`, `budget`, and `preferences` (e.g. `["shopping", "food", "culture"]`) are, the better the plans.
4. **Structured output (advanced)** – For even more reliable JSON, you can later switch to [Ollama’s structured output](https://docs.ollama.com/capabilities/structured-outputs) with a JSON schema once your API is stable.
