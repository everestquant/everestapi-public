# everestapi

Python SDK for the [EverestQuant](https://everestquant.ai) prediction tournament platform.

```bash
pip install everestapi
```

## Quickstart

```python
from everestapi import EverestAPI

api = EverestAPI(api_key="eiq_your_key")

# Browse the universe
universe = api.get_universe()

# Download training data
api.download_dataset(universe="futures", split="train", output_path="train.parquet")

# Submit predictions
api.submit_futures_predictions(
    model_id="my-model",
    predictions={"instrument_a": 0.5, "instrument_b": -0.3},
)

# Check scores
scores = api.get_scores(model_id="my-model", days=30)
```

Or set `EVEREST_API_KEY` as an environment variable and omit the constructor argument.

> **Handling your API key.** Shell history, terminal recordings, and CI logs may persist any value you `echo` or `print`. Copy the key via the clipboard rather than echoing it in a recorded session, and prefer storing it in a secrets manager or `.env` file (gitignored) over inlining in source.

## Two tournaments

| Tournament | Universe | Features | Frequency |
|------------|----------|----------|-----------|
| **Alps** (Equities) | Large-cap equities | Obfuscated fundamental + technical | Daily |
| **Himalayas** (Futures) | Global futures | Obfuscated cross-sectional + macro | Weekly |

```python
# Equities
api = EverestAPI(api_key="...", tournament="equities")
api.submit_predictions(model_id="my-eq-model", predictions=[...])

# Futures
api = EverestAPI(api_key="...", tournament="futures")
api.submit_futures_predictions(model_id="my-fut-model", predictions={...})
```

## Key features

### Data & diagnostics

```python
api.download_dataset(universe="futures", split="train")
api.download_benchmark(universe="futures", split="validation")
api.get_dataset_info(universe="futures")
api.get_diagnostics(model_id="my-model")
api.submit_validation_diagnostics(model_id="my-model", predictions=df)
```

### Serverless compute

```python
# Tier 1 — quick-train with built-in templates
job = api.quick_train(model="lightgbm", features="small", target="target_everest_20")

# Tier 2 — custom script on GPU
job = api.custom_train(script_path="train.py", gpu="A100", max_hours=2.0)

# Wait and download
result = api.wait_for_job(job["job_id"])
api.download_model(job["job_id"], output_path="model.pkl")
```

> **Pickle safety.** Trained models are returned as pickle files. `pickle.load` is RCE-equivalent: only load `.pkl` files from compute jobs you initiated yourself. Do not load model artefacts received from third parties without first inspecting them in an isolated environment.

### Staking (USDC)

```python
api.stake(model_id="my-model", amount_usdc=100.0, wallet_address="0x...")
api.get_stake_balance(model_id="my-model")
api.claim_payout(model_id="my-model", round_id="42")
```

### Local evaluation

```python
import pandas as pd

val = pd.read_parquet("futures_validation.parquet")
metrics = EverestAPI.evaluate(predictions, val, target="target_everest_20")
print(metrics)  # {"simple_corr": 0.023, "weighted_corr": 0.019, "per_difficulty": {...}}
```

### CLI

```bash
everestapi health
everestapi universe
everestapi submit --model my-model --file predictions.csv
```

## Registration

No API key needed to register:

```python
result = EverestAPI().register(name="my-agent", email="agent@example.com")
print(result["api_key"])  # shown once — save it
```

## Context manager

```python
with EverestAPI(api_key="...") as api:
    universe = api.get_universe()
    # connection pool cleaned up on exit
```

## Requirements

- Python 3.10+
- httpx >= 0.27

## Disclaimers

- **Not financial advice.** EverestQuant tournaments are prediction competitions. Nothing in this SDK or on the platform constitutes investment advice, a solicitation, or a recommendation to buy or sell any financial instrument.
- **Testnet / beta.** The staking system and compute platform are in beta. Smart contract addresses, API endpoints, and payout mechanics may change without notice.
- **API stability.** This SDK targets API v1. Breaking changes will be communicated via the platform changelog and will follow semver once the SDK reaches 1.0.
- **Data is obfuscated.** All features and instrument identifiers served by the API are obfuscated. Attempting to reverse-engineer or de-obfuscate data violates the platform terms of service.

## License

MIT — see [LICENSE](LICENSE).
