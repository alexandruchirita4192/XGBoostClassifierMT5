#property strict
#property version   "4.00"
#property description "EA MT5: XGBoost classifier trained in Python, exported to ONNX, run in Strategy Tester"
#property description "With trend filter, ATR volatility filter, and optional kill switch"

#include <Trade/Trade.mqh>

// IMPORTANT:
// 1) Copy the file ml_strategy_classifier_xgboost.onnx into the same folder as this .mq5.
// 2) Recompile the EA after copying.
#resource "ml_strategy_classifier_xgboost.onnx" as uchar ExtModel[]

input double InpLots                  = 0.10;      // InpLots: Fixed lot
input double InpEntryProbThreshold    = 0.60;      // InpEntryProbThreshold: Minimum probability threshold for BUY/SELL
input double InpMinProbGap            = 0.15;      // InpMinProbGap: Minimum difference between the best class and the next one
input bool   InpUseAtrStops           = true;      // InpUseAtrStops: Use SL/TP based on ATR
input double InpStopAtrMultiple       = 1.50;      // InpStopAtrMultiple: SL = ATR * multiplier
input double InpTakeAtrMultiple       = 2.00;      // InpTakeAtrMultiple: TP = ATR * multiplier
input int    InpMaxBarsInTrade        = 8;         // InpMaxBarsInTrade: Recommended to match horizon_bars from Python
input bool   InpCloseOnOppositeSignal = true;      // InpCloseOnOppositeSignal: Close on opposite signal
input bool   InpAllowLong             = true;      // InpAllowLong: Allow BUY
input bool   InpAllowShort            = true;      // InpAllowShort: Allow SELL

input bool   InpUseTrendFilter        = true;      // InpUseTrendFilter: Enable trend filter
input ENUM_TIMEFRAMES InpTrendTF      = PERIOD_H1; // InpTrendTF: Trend timeframe
input int    InpTrendMAPeriod         = 100;       // InpTrendMAPeriod: EMA period
input bool   InpTrendRequireSlope     = true;      // InpTrendRequireSlope: EMA must also have slope in signal direction
input bool   InpUseTrendDistanceFilter = false;    // InpUseTrendDistanceFilter: Require minimum distance from HTF EMA
input double InpTrendMinDistancePct    = 0.0010;   // InpTrendMinDistancePct: Minimum distance from EMA (0.001 = 0.1%)

input bool   InpUseAtrVolFilter        = true;     // InpUseAtrVolFilter: Enable ATR filter
input int    InpAtrVolLookback         = 50;       // InpAtrVolLookback: Number of bars for ATR distribution
input double InpAtrMinPercentile       = 0.25;     // InpAtrMinPercentile: Minimum ATR percentile in distribution (0..1)
input double InpAtrMaxPercentile       = 0.85;     // InpAtrMaxPercentile: Maximum ATR percentile in distribution (0..1)

input bool   InpUseKillSwitch                 = false; // InpUseKillSwitch: Enable kill switch
input int    InpKillSwitchLookbackTrades      = 8;     // InpKillSwitchLookbackTrades: Last N trades analyzed
input double InpKillSwitchMinWinRate          = 0.40;  // InpKillSwitchMinWinRate: Minimum accepted win rate
input double InpKillSwitchMinProfitFactor     = 0.95;  // InpKillSwitchMinProfitFactor: Minimum accepted profit factor
input int    InpKillSwitchConsecutiveLosses   = 4;     // InpKillSwitchConsecutiveLosses: Maximum consecutive losses
input int    InpKillSwitchPauseBars           = 96;    // InpKillSwitchPauseBars: Pause in bars after activation
input bool   InpKillSwitchFlatOnActivate      = true;  // InpKillSwitchFlatOnActivate: Close the current position when activated

input long   InpMagic                 = 26042026;  // InpMagic: Magic number
input bool   InpLog                   = false;     // InpLog: Main log
input bool   InpDebugLog              = false;     // InpDebugLog: Log on every new bar

const int FEATURE_COUNT = 10;
const int CLASS_COUNT   = 3; // class order: SELL, FLAT, BUY
const long EXT_INPUT_SHAPE[]  = {1, FEATURE_COUNT};
const long EXT_LABEL_SHAPE[]  = {1};
const long EXT_PROBA_SHAPE[]  = {1, CLASS_COUNT};

CTrade trade;
long g_model_handle = INVALID_HANDLE;
int  g_trend_ma_handle = INVALID_HANDLE;

datetime g_last_bar_time = 0;
int g_bars_in_trade = 0;

bool g_kill_switch_active = false;
int  g_kill_switch_pause_remaining = 0;
int  g_consecutive_losses = 0;
double g_recent_closed_profits[];
int g_last_history_deals_total = 0;

enum SignalDirection
  {
   SIGNAL_SELL = -1,
   SIGNAL_FLAT =  0,
   SIGNAL_BUY  =  1
  };

bool IsNewBar()
  {
   datetime current_bar_time = iTime(_Symbol, _Period, 0);
   if(current_bar_time == 0)
      return false;

   if(g_last_bar_time == 0)
     {
      g_last_bar_time = current_bar_time;
      return false;
     }

   if(current_bar_time != g_last_bar_time)
     {
      g_last_bar_time = current_bar_time;
      return true;
     }
   return false;
  }

double Mean(const double &arr[], int start_shift, int count)
  {
   double sum = 0.0;
   for(int i = start_shift; i < start_shift + count; i++)
      sum += arr[i];
   return sum / count;
  }

double StdDev(const double &arr[], int start_shift, int count)
  {
   double m = Mean(arr, start_shift, count);
   double s = 0.0;
   for(int i = start_shift; i < start_shift + count; i++)
     {
      double d = arr[i] - m;
      s += d * d;
     }
   return MathSqrt(s / MathMax(count - 1, 1));
  }

double CalcATR(const MqlRates &rates[], int start_shift, int period)
  {
   double sum_tr = 0.0;
   for(int i = start_shift; i < start_shift + period; i++)
     {
      double high = rates[i].high;
      double low = rates[i].low;
      double prev_close = rates[i + 1].close;
      double tr1 = high - low;
      double tr2 = MathAbs(high - prev_close);
      double tr3 = MathAbs(low - prev_close);
      double tr = MathMax(tr1, MathMax(tr2, tr3));
      sum_tr += tr;
     }
   return sum_tr / period;
  }

double GetPercentileFromArray(const double &arr[], int count, double q)
  {
   if(count <= 0)
      return 0.0;

   if(q <= 0.0)
      q = 0.0;
   if(q >= 1.0)
      q = 1.0;

   double tmp[];
   ArrayResize(tmp, count);
   for(int i = 0; i < count; i++)
      tmp[i] = arr[i];

   ArraySort(tmp);

   double pos = q * (count - 1);
   int lo = (int)MathFloor(pos);
   int hi = (int)MathCeil(pos);

   if(lo == hi)
      return tmp[lo];

   double w = pos - lo;
   return tmp[lo] * (1.0 - w) + tmp[hi] * w;
  }

bool BuildFeatureVector(matrixf &features, double &atr14)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   if(CopyRates(_Symbol, _Period, 0, 80, rates) < 40)
     {
      if(InpLog)
         Print("Not enough bars for features.");
      return false;
     }

   double closes[];
   ArrayResize(closes, ArraySize(rates));
   ArraySetAsSeries(closes, true);
   for(int i = 0; i < ArraySize(rates); i++)
      closes[i] = rates[i].close;

   int s = 1;

   double ret_1  = (closes[s] / closes[s + 1]) - 1.0;
   double ret_3  = (closes[s] / closes[s + 3]) - 1.0;
   double ret_5  = (closes[s] / closes[s + 5]) - 1.0;
   double ret_10 = (closes[s] / closes[s + 10]) - 1.0;

   double one_bar_returns[];
   ArrayResize(one_bar_returns, 30);
   for(int i = 0; i < 30; i++)
      one_bar_returns[i] = (closes[s + i] / closes[s + i + 1]) - 1.0;

   double vol_10 = StdDev(one_bar_returns, 0, 10);
   double vol_20 = StdDev(one_bar_returns, 0, 20);

   double sma_10 = Mean(closes, s, 10);
   double sma_20 = Mean(closes, s, 20);
   if(sma_10 == 0.0 || sma_20 == 0.0)
      return false;

   double dist_sma_10 = (closes[s] / sma_10) - 1.0;
   double dist_sma_20 = (closes[s] / sma_20) - 1.0;

   double mean_20 = Mean(closes, s, 20);
   double std_20  = StdDev(closes, s, 20);
   double zscore_20 = 0.0;
   if(std_20 > 0.0)
      zscore_20 = (closes[s] - mean_20) / std_20;

   atr14 = CalcATR(rates, s, 14);

   features.Resize(1, FEATURE_COUNT);
   features[0][0] = (float)ret_1;
   features[0][1] = (float)ret_3;
   features[0][2] = (float)ret_5;
   features[0][3] = (float)ret_10;
   features[0][4] = (float)vol_10;
   features[0][5] = (float)vol_20;
   features[0][6] = (float)dist_sma_10;
   features[0][7] = (float)dist_sma_20;
   features[0][8] = (float)zscore_20;
   features[0][9] = (float)atr14;

   if(InpDebugLog && InpLog)
     {
      PrintFormat(
         "FEATURES ret1=%.8f ret3=%.8f ret5=%.8f ret10=%.8f vol10=%.8f vol20=%.8f dist10=%.8f dist20=%.8f z20=%.8f atr14=%.8f",
         ret_1, ret_3, ret_5, ret_10, vol_10, vol_20, dist_sma_10, dist_sma_20, zscore_20, atr14
      );
     }

   return true;
  }

bool PredictClassProbabilities(double &pSell, double &pFlat, double &pBuy, double &atr14)
  {
   matrixf x;
   if(!BuildFeatureVector(x, atr14))
      return false;

   long predicted_label[1];
   matrixf probs;
   probs.Resize(1, CLASS_COUNT);

   if(!OnnxRun(g_model_handle, 0, x, predicted_label, probs))
     {
      if(InpLog)
         Print("OnnxRun failed. Error=", GetLastError());
      return false;
     }

   pSell = probs[0][0];
   pFlat = probs[0][1];
   pBuy  = probs[0][2];

   if(InpDebugLog && InpLog)
      PrintFormat("RAW ONNX label=%d probs: sell=%.6f flat=%.6f buy=%.6f",
                  predicted_label[0], pSell, pFlat, pBuy);

   return true;
  }

SignalDirection SignalFromProbabilities(double pSell, double pFlat, double pBuy)
  {
   double best = pFlat;
   double second = -1.0;
   SignalDirection signal = SIGNAL_FLAT;

   if(pBuy >= pSell && pBuy > best)
     {
      second = MathMax(best, pSell);
      best = pBuy;
      signal = SIGNAL_BUY;
     }
   else if(pSell > pBuy && pSell > best)
     {
      second = MathMax(best, pBuy);
      best = pSell;
      signal = SIGNAL_SELL;
     }
   else
     {
      second = MathMax(pBuy, pSell);
      signal = SIGNAL_FLAT;
     }

   double gap = best - second;

   if(signal == SIGNAL_BUY)
     {
      if(!InpAllowLong)
         return SIGNAL_FLAT;
      if(pBuy < InpEntryProbThreshold || gap < InpMinProbGap)
         return SIGNAL_FLAT;
      return SIGNAL_BUY;
     }

   if(signal == SIGNAL_SELL)
     {
      if(!InpAllowShort)
         return SIGNAL_FLAT;
      if(pSell < InpEntryProbThreshold || gap < InpMinProbGap)
         return SIGNAL_FLAT;
      return SIGNAL_SELL;
     }

   return SIGNAL_FLAT;
  }

bool GetTrendFilterValues(double &htf_close_1, double &ema_1, double &ema_2)
  {
   if(!InpUseTrendFilter)
      return true;

   if(g_trend_ma_handle == INVALID_HANDLE)
      return false;

   htf_close_1 = iClose(_Symbol, InpTrendTF, 1);
   if(htf_close_1 == 0.0)
      return false;

   double ema_buf[];
   ArraySetAsSeries(ema_buf, true);

   if(CopyBuffer(g_trend_ma_handle, 0, 1, 2, ema_buf) < 2)
      return false;

   ema_1 = ema_buf[0];
   ema_2 = ema_buf[1];
   return true;
  }

bool TrendAllows(SignalDirection signal)
  {
   if(!InpUseTrendFilter)
      return true;

   if(signal == SIGNAL_FLAT)
      return true;

   double htf_close_1 = 0.0;
   double ema_1 = 0.0;
   double ema_2 = 0.0;

   if(!GetTrendFilterValues(htf_close_1, ema_1, ema_2))
     {
      if(InpLog)
         Print("Trend filter: nu pot citi datele HTF.");
      return false;
     }

   bool slope_up   = (ema_1 > ema_2);
   bool slope_down = (ema_1 < ema_2);

   double distance_pct = 0.0;
   if(ema_1 != 0.0)
      distance_pct = MathAbs(htf_close_1 - ema_1) / ema_1;

   if(InpDebugLog && InpLog)
      PrintFormat(
         "TREND htf_close=%.5f ema1=%.5f ema2=%.5f slope_up=%d slope_down=%d distance_pct=%.6f",
         htf_close_1, ema_1, ema_2, slope_up, slope_down, distance_pct
      );

   if(InpUseTrendDistanceFilter && distance_pct < InpTrendMinDistancePct)
      return false;

   if(signal == SIGNAL_BUY)
     {
      if(htf_close_1 <= ema_1)
         return false;
      if(InpTrendRequireSlope && !slope_up)
         return false;
      return true;
     }

   if(signal == SIGNAL_SELL)
     {
      if(htf_close_1 >= ema_1)
         return false;
      if(InpTrendRequireSlope && !slope_down)
         return false;
      return true;
     }

   return true;
  }

bool AtrVolatilityAllows(double current_atr14)
  {
   if(!InpUseAtrVolFilter)
      return true;

   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int need_bars = InpAtrVolLookback + 20;
   if(CopyRates(_Symbol, _Period, 0, need_bars, rates) < need_bars)
     {
      if(InpLog)
         Print("ATR volatility filter: nu sunt suficiente bare.");
      return false;
     }

   double atr_values[];
   ArrayResize(atr_values, InpAtrVolLookback);

   int s = 1;
   for(int i = 0; i < InpAtrVolLookback; i++)
      atr_values[i] = CalcATR(rates, s + i, 14);

   double atr_min = GetPercentileFromArray(atr_values, InpAtrVolLookback, InpAtrMinPercentile);
   double atr_max = GetPercentileFromArray(atr_values, InpAtrVolLookback, InpAtrMaxPercentile);

   if(InpDebugLog && InpLog)
      PrintFormat("ATR FILTER current=%.6f min=%.6f max=%.6f", current_atr14, atr_min, atr_max);

   if(current_atr14 < atr_min)
      return false;
   if(current_atr14 > atr_max)
      return false;

   return true;
  }

bool HasOpenPosition(long &pos_type, double &pos_price)
  {
   if(!PositionSelect(_Symbol))
      return false;

   if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic)
      return false;

   pos_type = (long)PositionGetInteger(POSITION_TYPE);
   pos_price = PositionGetDouble(POSITION_PRICE_OPEN);
   return true;
  }

void CloseOpenPosition()
  {
   if(PositionSelect(_Symbol) && (long)PositionGetInteger(POSITION_MAGIC) == InpMagic)
     {
      bool ok = trade.PositionClose(_Symbol);
      if(!ok && InpLog)
         PrintFormat("PositionClose failed. retcode=%d lastError=%d",
                     trade.ResultRetcode(), GetLastError());
     }
  }

void PushClosedTradeProfit(double value)
  {
   int size = ArraySize(g_recent_closed_profits);
   ArrayResize(g_recent_closed_profits, size + 1);
   g_recent_closed_profits[size] = value;

   if(ArraySize(g_recent_closed_profits) > InpKillSwitchLookbackTrades)
     {
      for(int i = 1; i < ArraySize(g_recent_closed_profits); i++)
         g_recent_closed_profits[i - 1] = g_recent_closed_profits[i];
      ArrayResize(g_recent_closed_profits, InpKillSwitchLookbackTrades);
     }
  }

void ActivateKillSwitch(string reason)
  {
   if(!InpUseKillSwitch)
      return;

   g_kill_switch_active = true;
   g_kill_switch_pause_remaining = InpKillSwitchPauseBars;

   if(InpLog)
      PrintFormat("KILL SWITCH ACTIVATED: %s | pause_bars=%d", reason, g_kill_switch_pause_remaining);

   if(InpKillSwitchFlatOnActivate)
      CloseOpenPosition();
  }

void DecrementKillSwitchPause()
  {
   if(!g_kill_switch_active)
      return;

   if(g_kill_switch_pause_remaining > 0)
     {
      g_kill_switch_pause_remaining--;
      if(InpDebugLog && InpLog)
         PrintFormat("KillSwitch pause remaining: %d bars", g_kill_switch_pause_remaining);
     }

   if(g_kill_switch_pause_remaining <= 0)
     {
      g_kill_switch_active = false;
      g_consecutive_losses = 0;
      ArrayResize(g_recent_closed_profits, 0);
      if(InpLog)
         Print("KILL SWITCH DEACTIVATED.");
     }
  }

void EvaluateKillSwitch()
  {
   if(!InpUseKillSwitch)
      return;

   if(g_kill_switch_active)
      return;

   if(g_consecutive_losses >= InpKillSwitchConsecutiveLosses)
     {
      ActivateKillSwitch(StringFormat("consecutive_losses=%d", g_consecutive_losses));
      return;
     }

   int n = ArraySize(g_recent_closed_profits);
   if(n < InpKillSwitchLookbackTrades)
      return;

   int wins = 0;
   double gross_profit = 0.0;
   double gross_loss_abs = 0.0;

   for(int i = 0; i < n; i++)
     {
      double p = g_recent_closed_profits[i];
      if(p > 0.0)
        {
         wins++;
         gross_profit += p;
        }
      else if(p < 0.0)
        {
         gross_loss_abs += MathAbs(p);
        }
     }

   double win_rate = (double)wins / (double)n;
   double profit_factor = (gross_loss_abs > 0.0 ? gross_profit / gross_loss_abs : 999.0);

   if(InpDebugLog && InpLog)
      PrintFormat("KillSwitch stats: n=%d win_rate=%.3f profit_factor=%.3f consec_losses=%d",
                  n, win_rate, profit_factor, g_consecutive_losses);

   if(win_rate < InpKillSwitchMinWinRate)
     {
      ActivateKillSwitch(StringFormat("win_rate %.3f < %.3f", win_rate, InpKillSwitchMinWinRate));
      return;
     }

   if(profit_factor < InpKillSwitchMinProfitFactor)
     {
      ActivateKillSwitch(StringFormat("profit_factor %.3f < %.3f", profit_factor, InpKillSwitchMinProfitFactor));
      return;
     }
  }

void UpdateClosedTradeStats()
  {
   if(!InpUseKillSwitch)
      return;

   if(!HistorySelect(0, TimeCurrent()))
      return;

   int total = HistoryDealsTotal();
   if(total <= g_last_history_deals_total)
      return;

   for(int i = g_last_history_deals_total; i < total; i++)
     {
      ulong deal_ticket = HistoryDealGetTicket(i);
      if(deal_ticket == 0)
         continue;

      string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
      long magic    = HistoryDealGetInteger(deal_ticket, DEAL_MAGIC);
      long entry    = HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);

      if(symbol != _Symbol)
         continue;
      if(magic != InpMagic)
         continue;
      if(entry != DEAL_ENTRY_OUT)
         continue;

      double profit     = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
      double swap       = HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
      double commission = HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
      double net = profit + swap + commission;

      PushClosedTradeProfit(net);

      if(net < 0.0)
         g_consecutive_losses++;
      else if(net > 0.0)
         g_consecutive_losses = 0;

      if(InpLog)
         PrintFormat("Closed trade detected: net=%.2f consec_losses=%d", net, g_consecutive_losses);
     }

   g_last_history_deals_total = total;
   EvaluateKillSwitch();
  }

void OpenTrade(SignalDirection signal, double atr14)
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   double min_stop = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   double sl_dist = MathMax(atr14 * InpStopAtrMultiple, min_stop);
   double tp_dist = MathMax(atr14 * InpTakeAtrMultiple, min_stop);

   double sl = 0.0;
   double tp = 0.0;

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(20);

   bool ok = false;

   if(signal == SIGNAL_BUY)
     {
      if(InpUseAtrStops)
        {
         sl = ask - sl_dist;
         tp = ask + tp_dist;
        }
      ok = trade.Buy(InpLots, _Symbol, ask, sl, tp, "LGBM class buy");
      if(ok)
         g_bars_in_trade = 0;
      else if(InpLog)
         PrintFormat("BUY failed. retcode=%d lastError=%d ask=%.5f sl=%.5f tp=%.5f",
                     trade.ResultRetcode(), GetLastError(), ask, sl, tp);
     }
   else if(signal == SIGNAL_SELL)
     {
      if(InpUseAtrStops)
        {
         sl = bid + sl_dist;
         tp = bid - tp_dist;
        }
      ok = trade.Sell(InpLots, _Symbol, bid, sl, tp, "LGBM class sell");
      if(ok)
         g_bars_in_trade = 0;
      else if(InpLog)
         PrintFormat("SELL failed. retcode=%d lastError=%d bid=%.5f sl=%.5f tp=%.5f",
                     trade.ResultRetcode(), GetLastError(), bid, sl, tp);
     }
  }

void ManageExistingPosition(SignalDirection signal)
  {
   long pos_type;
   double pos_price;
   if(!HasOpenPosition(pos_type, pos_price))
      return;

   g_bars_in_trade++;
   bool should_close = false;

   if(InpCloseOnOppositeSignal)
     {
      if(pos_type == POSITION_TYPE_BUY  && signal == SIGNAL_SELL)
         should_close = true;
      if(pos_type == POSITION_TYPE_SELL && signal == SIGNAL_BUY)
         should_close = true;
     }

   if(!should_close && g_bars_in_trade >= InpMaxBarsInTrade)
      should_close = true;

   if(should_close)
      CloseOpenPosition();
  }

int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagic);

   g_model_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(g_model_handle == INVALID_HANDLE)
     {
      if(InpLog)
         Print("OnnxCreateFromBuffer failed. Error=", GetLastError());
      return INIT_FAILED;
     }

   if(!OnnxSetInputShape(g_model_handle, 0, EXT_INPUT_SHAPE))
     {
      if(InpLog)
         Print("OnnxSetInputShape failed. Error=", GetLastError());
      OnnxRelease(g_model_handle);
      g_model_handle = INVALID_HANDLE;
      return INIT_FAILED;
     }

   if(!OnnxSetOutputShape(g_model_handle, 0, EXT_LABEL_SHAPE))
     {
      if(InpLog)
         Print("OnnxSetOutputShape(label) failed. Error=", GetLastError());
      OnnxRelease(g_model_handle);
      g_model_handle = INVALID_HANDLE;
      return INIT_FAILED;
     }

   if(!OnnxSetOutputShape(g_model_handle, 1, EXT_PROBA_SHAPE))
     {
      if(InpLog)
         Print("OnnxSetOutputShape(probabilities) failed. Error=", GetLastError());
      OnnxRelease(g_model_handle);
      g_model_handle = INVALID_HANDLE;
      return INIT_FAILED;
     }

   if(InpUseTrendFilter)
     {
      g_trend_ma_handle = iMA(_Symbol, InpTrendTF, InpTrendMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
      if(g_trend_ma_handle == INVALID_HANDLE)
        {
         if(InpLog)
            Print("Trend filter iMA handle failed. Error=", GetLastError());
         OnnxRelease(g_model_handle);
         g_model_handle = INVALID_HANDLE;
         return INIT_FAILED;
        }
     }

   if(HistorySelect(0, TimeCurrent()))
      g_last_history_deals_total = HistoryDealsTotal();
   else
      g_last_history_deals_total = 0;

   ArrayResize(g_recent_closed_profits, 0);
   g_consecutive_losses = 0;
   g_kill_switch_active = false;
   g_kill_switch_pause_remaining = 0;

   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   if(g_model_handle != INVALID_HANDLE)
     {
      OnnxRelease(g_model_handle);
      g_model_handle = INVALID_HANDLE;
     }

   if(g_trend_ma_handle != INVALID_HANDLE)
     {
      IndicatorRelease(g_trend_ma_handle);
      g_trend_ma_handle = INVALID_HANDLE;
     }
  }

void OnTick()
  {
   if(!IsNewBar())
      return;

   UpdateClosedTradeStats();
   DecrementKillSwitchPause();

   if(g_kill_switch_active)
     {
      if(InpDebugLog && InpLog)
         Print("KillSwitch active -> skip new entries.");
      return;
     }

   double pSell = 0.0;
   double pFlat = 0.0;
   double pBuy  = 0.0;
   double atr14 = 0.0;

   if(!PredictClassProbabilities(pSell, pFlat, pBuy, atr14))
      return;

   if(!AtrVolatilityAllows(atr14))
     {
      if(InpDebugLog && InpLog)
         PrintFormat("ATR volatility filter blocked entry. atr14=%.6f", atr14);
      ManageExistingPosition(SIGNAL_FLAT);
      return;
     }

   SignalDirection raw_signal = SignalFromProbabilities(pSell, pFlat, pBuy);
   SignalDirection filtered_signal = raw_signal;

   if(!TrendAllows(raw_signal))
      filtered_signal = SIGNAL_FLAT;

   if(InpDebugLog && InpLog)
     {
      PrintFormat(
         "Probabilities sell=%.4f flat=%.4f buy=%.4f entry_prob=%.4f min_gap=%.4f raw_signal=%d filtered_signal=%d atr14=%.5f",
         pSell, pFlat, pBuy, InpEntryProbThreshold, InpMinProbGap, raw_signal, filtered_signal, atr14
      );
     }

   ManageExistingPosition(filtered_signal);

   long pos_type;
   double pos_price;
   if(HasOpenPosition(pos_type, pos_price))
      return;

   if(filtered_signal == SIGNAL_BUY || filtered_signal == SIGNAL_SELL)
      OpenTrade(filtered_signal, atr14);
  }

double OnTester() {
  double profit = TesterStatistics(STAT_PROFIT);
  double pf = TesterStatistics(STAT_PROFIT_FACTOR);
  double recovery = TesterStatistics(STAT_RECOVERY_FACTOR);
  double dd_percent = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
  double trades = TesterStatistics(STAT_TRADES);

  // Penalty if there are too few transactions
  double trade_penalty = 1.0;
  if (trades < 20)
    trade_penalty = 0.25;
  else if (trades < 50)
    trade_penalty = 0.60;

  // Robust score, not only brut profit
  double score = 0.0;

  if (dd_percent >= 0.0)
    score =
        (profit * MathMax(pf, 0.01) * MathMax(recovery, 0.01) * trade_penalty) /
        (1.0 + dd_percent);

  return score;
}
