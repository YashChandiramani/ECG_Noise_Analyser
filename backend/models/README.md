# Model weights

The supplied notebook defines the `Paper1DCNN` architecture but does not contain a
`torch.save(...)` checkpoint export in the searchable notebook content.

After training, export its state dictionary from the notebook, for example:

```python
torch.save(model.state_dict(), "paper_1d_cnn.pt")
```

Then place the resulting file here:

`backend/models/paper_1d_cnn.pt`

The current API intentionally refuses to invent predictions when the trained
weights are absent.

Important: the notebook contains both a 5-class MIT-BIH ECG dataset loader and a
separate binary synthetic clinical-noise pipeline. A checkpoint must match the
architecture, number of classes, labels, and preprocessing used at inference.
Do not mix a 5-class arrhythmia checkpoint with a binary noise UI.
