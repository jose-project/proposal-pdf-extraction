# EB3 Proposal Rate Extractor API

FastAPI-based REST API for extracting insurance plan rates from proposal PDFs using LLM.

## Installation

```bash
pip install -r requirements.txt
```

## Running the API

```bash
python api.py
```

Or using uvicorn directly:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### `POST /extract`

Extract rates from a single PDF file.

**Parameters:**
- `pdf` (file, required): PDF proposal file
- This endpoint uses default configuration from `scripts/constants.py` (chunking, limits, concurrency)

**Response:**
```json
{
  "file": "proposal.pdf",
  "carrier": "Blue Cross Blue Shield of Illinois",
  "plans": [
    {
      "plan_name": "Silver PPO",
      "plan_id": "P503PPO",
      "rate_structure": "4_tier",
      "rates": {
        "employee_only": 1407.14,
        "employee_spouse": 2814.28,
        "employee_child": 2603.21,
        "family": 4010.35
      },
      "source_pages": [9, 10]
    }
  ],
  "status": "success"
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/extract" \
  -F "pdf=@source/proposals/CityKids Dental Renewal 2026 BCBSiL.pdf"
```

**Example using Python requests:**
```python
import requests

url = "http://localhost:8000/extract"
with open("source/proposals/CityKids Dental Renewal 2026 BCBSiL.pdf", "rb") as f:
    files = {"pdf": f}
    response = requests.post(url, files=files)
    print(response.json())
```

### `POST /extract-batch`

Extract rates from multiple PDF files in batch.

**Parameters:**
- `pdfs` (files, required): Multiple PDF proposal files
- This endpoint uses default configuration from `scripts/constants.py` (chunking, limits, concurrency)

**Response:**
```json
{
  "results": [...],
  "errors": [...],
  "total": 5,
  "successful": 4,
  "failed": 1
}
```

### `GET /health`

Check API health status.

**Response:**
```json
{
  "status": "healthy"
}
```

### `GET /`

Get API information and available endpoints.

## Rate Structures

The API supports the following rate structure types:

- `2_tier`: Employee Only, Family
- `3_tier`: Employee Only, Employee + 1, Employee + 2 or more
- `4_tier`: Employee Only, Employee + Spouse, Employee + Child, Family
- `5_tier`: Employee Only, Employee + Spouse, Employee + Child, Employee + 2 or more Children, Family
- `8_tier`: Complex family structures (8 tiers)
- `aca_age`: Age-banded rates (ages <15 to 64+)
- `age_band_5`: 5-year age bands (<20, 20-24, ..., 80+)
- `age_band_10`: 10-year age bands (<20, 20-29, ..., 90+)
- `esc_5_year`: Employee/Spouse/Children with 5-year age bands
- `esc_10_year`: Employee/Spouse/Children with 10-year age bands
- `4_tier_5_year`: 4-tier structure with 5-year age bands
- `4_tier_10_year`: 4-tier structure with 10-year age bands
- `3_tier_age_band`: 3-tier with age bands (0-18, 19-20, 21+)
- `2_tier_age_band`: 2-tier with age bands (0-20, 21+)

## Configuration

The API uses a remote LLM endpoint. Update the endpoint in `scripts/remote_llm.py` if needed:

```python
class RemoteLLM:
    def __init__(self, endpoint: str = "http://143.110.210.212/v1/chat/completions"):
        self.endpoint = endpoint
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (e.g., non-PDF file)
- `500`: Server error (e.g., LLM processing error)

Error responses include details about what went wrong.

## Notes

- Large PDFs are processed in chunks to manage token limits
- Temporary files are automatically cleaned up
- The API uses async processing for better performance
- All rate values are normalized to floats/nulls

