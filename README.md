# MT5 + XGBoost + ONNX

Contine:
- `train_mt5_xgboost_classifier.py` — script Python pentru training, split cronologic train/test, etichetare pe 3 clase si export ONNX
- `MT5_XGBoost_Classifier_ONNX_Strategy.mq5` — EA MQL5 pentru Strategy Tester si rulare in MT5

## 1. Pachete Python necesare

In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install MetaTrader5 pandas numpy scikit-learn xgboost onnxmltools onnx onnxconverter-common
```

## 2. Comenzi recomandate

```powershell
python train_mt5_xgboost_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 8 --train-ratio 0.70 --output-dir output_xgb_h8
python train_mt5_xgboost_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 4 --train-ratio 0.70 --output-dir output_xgb_h4
python train_mt5_xgboost_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 12 --train-ratio 0.70 --output-dir output_xgb_h12
```

## 3. Fisiere rezultate

In folderul de output vei avea, intre altele:
- `ml_strategy_classifier_xgboost.onnx`
- `model_metadata.json`
- `run_in_mt5.txt`
- `train_predictions_snapshot.csv`
- `test_predictions_snapshot.csv`

## 4. Cum rulezi in MT5

1. Copiezi `ml_strategy_classifier_xgboost.onnx` in acelasi folder cu `MT5_XGBoost_Classifier_ONNX_Strategy.mq5`
2. Recompilezi EA-ul in MetaEditor
3. Deschizi `run_in_mt5.txt`
4. In Strategy Tester setezi exact fereastra `TEST UTC`
5. Introduci in inputuri valorile recomandate

## 5. Set de plecare pentru filtre in EA

Porneste de la benchmark-ul tau actual:

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

## 6. Observatie importanta

Ca si la LightGBM, primul output ONNX este eticheta (`int64`), iar al doilea este tensorul de probabilitati. EA-ul este deja scris pentru acest format.
