import { useMemo } from "react";
import { movingAverage, rsi, totalReturn, annualizedVolatility, dailyReturns } from "../utils/calculations";
import type { ChartPoint } from "../hooks/useStockData";

interface MomentumSignalsProps {
  chartData: ChartPoint[];
}

function Badge({ label, value, variant }: { label: string; value: string; variant: "green" | "yellow" | "red" }) {
  const bg = variant === "green" ? "bg-profit/20 text-profit" : variant === "yellow" ? "bg-amber-500/20 text-amber-400" : "bg-loss/20 text-loss";
  return (
    <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full font-mono text-sm border ${bg} border-current/30`}>
      <span className="text-gray-500">{label}:</span>
      <span>{value}</span>
    </div>
  );
}

export default function MomentumSignals({ chartData }: MomentumSignalsProps) {
  const signals = useMemo(() => {
    if (chartData.length < 200) return null;
    const closes = chartData.map((d) => d.close);
    const ma200 = movingAverage(closes, 200);
    const ma5vol = chartData.slice(-5).reduce((s, d) => s + d.volume, 0) / 5;
    const ma20vol = chartData.slice(-20).reduce((s, d) => s + d.volume, 0) / 20;

    const price = closes[closes.length - 1] ?? 0;
    const avg200 = ma200[ma200.length - 1];
    const trend: "Bullish" | "Bearish" | "Neutral" =
      avg200 != null ? (price > avg200 ? "Bullish" : price < avg200 ? "Bearish" : "Neutral") : "Neutral";

    const ret3m = chartData.length >= 63 ? totalReturn(closes.slice(-63)) : 0;
    const momentum: "Strong" | "Weak" | "Negative" =
      ret3m > 0.1 ? "Strong" : ret3m >= 0 ? "Weak" : "Negative";

    const allRets = dailyReturns(closes);
    const rets20d = allRets.slice(-20);
    const vol20d = annualizedVolatility(rets20d);
    const vol1y = annualizedVolatility(allRets);
    const volRegime: "High Vol" | "Normal" | "Low Vol" =
      vol20d > vol1y * 1.2 ? "High Vol" : vol20d < vol1y * 0.8 ? "Low Vol" : "Normal";

    const rsiVal = rsi(closes);
    const rsiSignal: "Overbought" | "Neutral" | "Oversold" =
      rsiVal > 70 ? "Overbought" : rsiVal < 30 ? "Oversold" : "Neutral";

    const volRatio = ma20vol > 0 ? ma5vol / ma20vol : 1;
    const volumeSignal: "High Volume" | "Normal" | "Low Volume" =
      volRatio > 1.2 ? "High Volume" : volRatio < 0.8 ? "Low Volume" : "Normal";

    return { trend, momentum, volRegime, rsiVal, rsiSignal, volumeSignal };
  }, [chartData]);

  if (!signals) return null;

  const trendVariant = signals.trend === "Bullish" ? "green" : signals.trend === "Bearish" ? "red" : "yellow";
  const momVariant = signals.momentum === "Strong" ? "green" : signals.momentum === "Negative" ? "red" : "yellow";
  const volRegVariant = signals.volRegime === "Low Vol" ? "green" : signals.volRegime === "High Vol" ? "red" : "yellow";
  const rsiVariant = signals.rsiSignal === "Oversold" ? "green" : signals.rsiSignal === "Overbought" ? "red" : "yellow";
  const volSigVariant = signals.volumeSignal === "High Volume" ? "green" : signals.volumeSignal === "Low Volume" ? "red" : "yellow";

  return (
    <div className="bg-dark-card border border-dark-border rounded-lg p-6">
      <h3 className="text-accent font-mono text-lg mb-4">Momentum Signals</h3>
      <div className="flex flex-wrap gap-3">
        <Badge label="Trend" value={signals.trend} variant={trendVariant} />
        <Badge label="Momentum" value={signals.momentum} variant={momVariant} />
        <Badge label="Volatility Regime" value={signals.volRegime} variant={volRegVariant} />
        <Badge label={`RSI (${signals.rsiVal.toFixed(0)})`} value={signals.rsiSignal} variant={rsiVariant} />
        <Badge label="Volume" value={signals.volumeSignal} variant={volSigVariant} />
      </div>
    </div>
  );
}
