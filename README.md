# Travel AI Backend

Python backend for generating trip itineraries. Uses **Ollama** (local) to produce 3 itinerary options and returns JSON matching the frontend spec.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ollama (local AI) – required for real itineraries

The app uses **Ollama** to generate itineraries. If Ollama is not running or fails, the API returns **503** with a `detail` message so the frontend can show "API failed" instead of generic data.

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

## API Endpoints

Base URL: frontend uses `VITE_ITINERARY_API_BASE` (default `http://127.0.0.1:8000`). All routes also under `/api/...` where noted.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/flights/track` | Track flight by number (e.g. `{"flight_number": "UA1234"}`). Requires `AVIATION_STACK_API_KEY`. |
| POST | `/itinerary/generate` | Generate 3 AI itinerary options. Returns **503** with a `detail` message if Ollama fails (no generic fallback). |
| POST | `/itinerary/plan-with-picks` | Build one itinerary from user picks (labels + optional Google Maps URLs). |
| POST | `/itinerary/quote` | Itemized quote + points optimization for a selected option. Optional `num_persons` in body. |
| POST | `/itinerary/trip-document` | Full trip document (itinerary, currency, language, emergency contacts). |
| GET | `/destinations` | Curated destinations for inspiration. |
| GET | `/destinations/trending` | Trending destinations (ordered subset). |

**Itinerary generate** accepts: `origin`, `destination`, `start_date`, `end_date`, `budget`, `per_person_budget`, `num_persons`, `accommodation_type`, `pace`, `preferences` (e.g. `["culture","food","nature"]`), `disability`, `dietary`. All optional.

CORS is enabled for `http://localhost:5173`, `http://localhost:3000`, and `https://voyager-ai.netlify.app`. Use `ngrok-skip-browser-warning: true` header when calling behind ngrok.

---

## Building the best itinerary planner (what to do)

1. **Use a capable model** – Set `OLLAMA_MODEL=llama3.1:8b` (or a larger model if you have RAM) for richer itineraries.
2. **Keep Ollama running** – If Ollama is down or times out, the API returns **503** with a `detail` message (e.g. "Itinerary API failed: cannot reach Ollama..."). The frontend can show that message instead of generic data.
3. **Timeout** – Ollama can take as long as it needs; default is **5 minutes** (set `OLLAMA_TIMEOUT_SEC` in `.env`, e.g. `600` or `900` for 10–15 min). The frontend should use a **long request timeout** (e.g. 5+ minutes) when calling `/itinerary/generate`.
4. **503 after 2–3 minutes** – Check the response body `detail` message. If you're behind **ngrok**, the free tier may close long-running requests; try calling the API directly (e.g. `http://localhost:8000`) to confirm the backend completes. If Ollama returns an error, the `detail` will include the HTTP status or snippet from Ollama.
5. **Send clear preferences** – The more specific `origin`, `destination`, `budget`, and `preferences` are, the better the plans.
6. **Structured output (advanced)** – For even more reliable JSON, you can later switch to [Ollama’s structured output](https://docs.ollama.com/capabilities/structured-outputs) with a JSON schema.

---

## Security & environment

- **API keys** – Use environment variables; do not commit keys. Copy `.env.example` to `.env` and set:
  - `AVIATION_STACK_API_KEY` – [Aviation Stack](https://aviationstack.com/signup/free) (free tier: 100 req/month) for **POST /flights/track**.
  - Optional: `GOOGLE_MAPS_API_KEY` for Places/Geocoding; `OLLAMA_MODEL`; `OLLAMA_TIMEOUT_SEC` (seconds, default 300); `CORS_ORIGINS` (comma-separated).
- **Validation** – All request bodies are validated via Pydantic.
- **CORS** – Allowed origins are explicit; add more via `CORS_ORIGINS`. Header `ngrok-skip-browser-warning` is allowed for ngrok.

---

## Integrations checklist

| Integration | Purpose | This repo |
|-------------|---------|-----------|
| Flight tracking | Real-time flight status | Aviation Stack (`AVIATION_STACK_API_KEY`), **POST /flights/track** |
| Itinerary AI | Generate plans | Ollama (local), **POST /itinerary/generate** |
| Places / Geocoding | Addresses, Maps URLs | Placeholder URLs; set `GOOGLE_MAPS_API_KEY` to enrich (optional) |
| Destinations | Inspiration / curation | Built-in list + optional Hugging Face `DeepNLP/travel-ai-agent` (`DESTINATIONS_USE_HF=1`, `pip install datasets`) |
