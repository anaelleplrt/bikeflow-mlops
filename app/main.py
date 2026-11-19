from fastapi import FastAPI, HTTPException

app = FastAPI(title="BikeFlow")

app.state.model = None
app.state.model_version = None


@app.get("/")
def home():
    return {"message": "Welcome to BikeFlow"}


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Approved model not loaded"
        )

    return {
        "status": "ready",
        "model_version": app.state.model_version
    }
