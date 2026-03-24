import {
  dailyReturns,
  totalReturn,
  annualizedVolatility,
  sharpeRatio,
  sortinoRatio,
  maxDrawdown,
  varHistorical,
  skewness,
  kurtosis,
  calmarRatio,
} from "../utils/calculations";
import type { ChartPoint } from "../hooks/useStockData";

function MetricCard({
  label,
  value,
  type,
  tooltip,
}: {
  label: string;
  value: string | number;
  type: "good" | "bad" | "neutral";
  tooltip: string;
}) {
  const color = type === "good" ? "text-profit" : type === "bad" ? "text-loss" : "text-gray-400";
  return (
    <div className="bg-dark-card border border-dark-border rounded-lg p-4 relative group">
      <div className="flex items-center gap-1">
        <p className="text-gray-500 text-sm">{label}</p>
        <span
          className="text-gray-600 cursor-help text-xs"
          title={tooltip}
        >
          ℹ
        </span>
      </div>
      <p className={`font-mono text-lg ${color}`}>{value}</p>
    </div>
  );
}

interface QuantMetricsPanelProps {
  chartData: ChartPoint[];
}

export default function QuantMetricsPanel({ chartData }: QuantMetricsPanelProps) {
  const closes = chartData.map((d) => d.close);
  const rets = dailyReturns(closes);

  const totalRet = totalReturn(closes);
  const annVol = annualizedVolatility(rets);
  const sharpe = sharpeRatio(rets);
  const sortino = sortinoRatio(rets);
  const { maxDrawdown: mdd, troughDateIndex } = maxDrawdown(closes);
  const bestDay = rets.length > 0 ? Math.max(...rets) * 100 : 0;
  const worstDay = rets.length > 0 ? Math.min(...rets) * 100 : 0;
  const mddDate = chartData[troughDateIndex]?.date ?? "—";

  const var95 = varHistorical(rets, 5) * 100;
  const var99 = varHistorical(rets, 1) * 100;
  const skew = skewness(rets);
  const kurt = kurtosis(rets);
  const calmar = calmarRatio(closes);
  const pctPositive = rets.length > 0 ? (rets.filter((r) => r > 0).length / rets.length) * 100 : 0;

  const good = (v: number, higherIsBetter: boolean) =>
    higherIsBetter ? (v > 0 ? "good" : "neutral") : v > 0 ? "bad" : "neutral";

  return (
    <div className="bg-dark-card border border-dark-border rounded-lg p-6">
      <h3 className="text-accent font-mono text-lg mb-4">Quant Metrics</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <h4 className="text-gray-500 text-sm mb-3 font-mono">Performance</h4>
          <div className="space-y-3">
            <MetricCard
              label="Total Return (1Y)"
              value={`${(totalRet * 100).toFixed(2)}%`}
              type={good(totalRet, true)}
              tooltip="Total percentage return over the period"
            />
            <MetricCard
              label="Annualized Volatility"
              value={`${(annVol * 100).toFixed(2)}%`}
              type="neutral"
              tooltip="Standard deviation of daily returns × √252"
            />
            <MetricCard
              label="Sharpe Ratio"
              value={sharpe.toFixed(3)}
              type={good(sharpe, true)}
              tooltip="(Return - risk-free rate) / Volatility; higher = better risk-adjusted return"
            />
            <MetricCard
              label="Sortino Ratio"
              value={sortino.toFixed(3)}
              type={good(sortino, true)}
              tooltip="Like Sharpe but uses downside deviation only"
            />
            <MetricCard
              label="Max Drawdown"
              value={`${(mdd * 100).toFixed(2)}% (${mddDate})`}
              type="bad"
              tooltip="Largest peak-to-trough decline; date when trough occurred"
            />
            <MetricCard
              label="Best / Worst Day"
              value={`${bestDay.toFixed(2)}% / ${worstDay.toFixed(2)}%`}
              type="neutral"
              tooltip="Single best and worst daily return"
            />
          </div>
        </div>
        <div>
          <h4 className="text-gray-500 text-sm mb-3 font-mono">Risk</h4>
          <div className="space-y-3">
            <MetricCard
              label="VaR 95% (1-day)"
              value={`${var95.toFixed(2)}%`}
              type="bad"
              tooltip="5th percentile of daily returns; historical method"
            />
            <MetricCard
              label="VaR 99%"
              value={`${var99.toFixed(2)}%`}
              type="bad"
              tooltip="1st percentile of daily returns"
            />
            <MetricCard
              label="Skewness"
              value={skew.toFixed(3)}
              type="neutral"
              tooltip="Asymmetry of return distribution; negative = left tail"
            />
            <MetricCard
              label="Kurtosis"
              value={kurt.toFixed(3)}
              type="neutral"
              tooltip="Tail heaviness; >0 = fatter tails than normal"
            />
            <MetricCard
              label="Calmar Ratio"
              value={calmar.toFixed(3)}
              type={good(calmar, true)}
              tooltip="Annualized return / |Max Drawdown|"
            />
            <MetricCard
              label="% Positive Days"
              value={`${pctPositive.toFixed(1)}%`}
              type={good(pctPositive - 50, true)}
              tooltip="Fraction of trading days with positive return"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
