# MT5 + XGBoost + ONNX

Contains:
- `train_mt5_xgboost_classifier.py` — Python script for training, chronological train/test split, labeling on 3 classes and ONNX export
- `MT5_XGBoost_Classifier_ONNX_Strategy.mq5` — MQL5 EA for Strategy Tester and running in MT5

## 1. Required Python packages

In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install MetaTrader5 pandas numpy scikit-learn xgboost onnxmltools onnx onnxconverter-common
```

## 2. Recommended commands

```powershell
python train_mt5_xgboost_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 8 --train-ratio 0.70 --output-dir output_xgb_h8
python train_mt5_xgboost_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 4 --train-ratio 0.70 --output-dir output_xgb_h4
python train_mt5_xgboost_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 12 --train-ratio 0.70 --output-dir output_xgb_h12
```

## 3. Result files

In the output folder you will have, among others:
- `ml_strategy_classifier_xgboost.onnx`
- `model_metadata.json`
- `run_in_mt5.txt`
- `train_predictions_snapshot.csv`
- `test_predictions_snapshot.csv`

## 4. How to run in MT5

1. Copy `ml_strategy_classifier_xgboost.onnx` to the same folder as `MT5_XGBoost_Classifier_ONNX_Strategy.mq5`
2. Recompile the EA in MetaEditor
3. Open `run_in_mt5.txt`
4. In Strategy Tester set exactly the `TEST UTC` window
5. Enter the recommended values in the inputs

## 5. Starting set for filters in EA

Start from your current benchmark:

```text
InpEntryProbThreshold = 0.60
InpMinProbGap         = 0.15
InpUseTrendFilter     = true
InpTrendTF            = PERIOD_H1
InpTrendMAPeriod      = 100
InpTrendRequireSlope  = true
InpUseTrendDistanceFilter = false
InpUseAtrVolFilter    = true
InpAtrVolLookback     = 50
InpAtrMinPercentile   = 0.25
InpAtrMaxPercentile   = 0.85
InpUseKillSwitch      = false
```

## 6. Important note

Like LightGBM, the first ONNX output is the label (`int64`), and the second is the probability tensor. The EA is already written for this format.
