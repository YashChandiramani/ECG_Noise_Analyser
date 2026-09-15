# ECGSense — ECG Clinical Signal Dashboard

This project contains a React/Vite frontend and FastAPI inference scaffold based
on the supplied ECG notebook.

## What is implemented

- ECG CSV/TXT upload
- Waveform visualization
- Model branch selector
- 1D CNN inference endpoint
- ResNet-18 branch placeholder
- Responsive dashboard
- Explicit missing-weight/error states

## Important notebook detail

The notebook describes a multi-branch ECG framework with:
- Branch A: 1D CNN on raw time-series signals
- Branch B: CWT + ResNet-18 on 224x224 RGB scalograms
- Branch C: convolutional autoencoder / latent representation

The notebook also contains a `Paper1DCNN(num_classes=5)` architecture and a
separate synthetic binary clinical-noise generation routine. Those are not the
same label space. The frontend therefore does not hard-code "clean/noisy" as the
prediction of the 5-class checkpoint.

## Run frontend

```bash
cd frontend
npm install
npm run dev
```

## Run backend

Use Python 3.11/3.12 for the PyTorch environment if your installed PyTorch build
does not support your local Python version.

```bash
cd backend
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open the Vite URL.

## Add trained weights

Export a compatible checkpoint from the notebook:

```python
torch.save(model.state_dict(), "paper_1d_cnn.pt")
```

Put it at:

```text
backend/models/paper_1d_cnn.pt
```

The API will report that weights are missing rather than producing a fake result.

## Environment

To point the frontend at another API:

```text
VITE_API_URL=http://localhost:8000
```
