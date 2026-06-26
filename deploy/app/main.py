"""FastAPI app for the Kramer classifier deploy demo.

GET  /healthz   no auth, Fly healthcheck target
GET  /          HTML upload form, behind Basic Auth (+ Cloudflare Access in front)
POST /predict   image upload -> calibrated probability, behind Basic Auth
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from . import config, inference
from .auth import require_auth

_model_holder: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _model_holder["model"] = inference.load_model()
    yield
    _model_holder.clear()


app = FastAPI(title="Kramer Classifier Demo", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


_PAGE = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Kramer Classifier Demo</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; color: #222; }}
    .disclaimer {{ background: #fff3cd; border: 1px solid #ffe69c; padding: 12px 16px; border-radius: 6px; font-size: 0.9em; margin-bottom: 24px; }}
    button {{ background: #2563eb; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; }}
    button:disabled {{ background: #93c5fd; }}
    #result {{ margin-top: 24px; white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 0.85em; background: #f5f5f5; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Kramer Classifier Demo</h1>
  <div class="disclaimer">{config.DISCLAIMER}</div>
  <input type="file" id="image" accept="image/*" />
  <button id="submit">Classify</button>
  <div id="result"></div>
  <script>
    document.getElementById('submit').onclick = async () => {{
      const fileInput = document.getElementById('image');
      const resultEl = document.getElementById('result');
      if (!fileInput.files.length) {{
        resultEl.textContent = 'Choose an image first.';
        return;
      }}
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      resultEl.textContent = 'Running inference...';
      try {{
        const res = await fetch('/predict', {{ method: 'POST', body: formData }});
        const data = await res.json();
        resultEl.textContent = JSON.stringify(data, null, 2);
      }} catch (err) {{
        resultEl.textContent = 'Error: ' + err;
      }}
    }};
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index(_user: str = Depends(require_auth)):
    return _PAGE


@app.post("/predict")
async def predict(file: UploadFile = File(...), _user: str = Depends(require_auth)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    image_bytes = await file.read()
    model = _model_holder["model"]
    result = inference.predict(model, image_bytes)
    return JSONResponse(result)
