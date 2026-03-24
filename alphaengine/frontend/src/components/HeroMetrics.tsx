import type { ChartPoint, QuoteData } from "../hooks/useStockData";

function fmtMarketCap(v: number): string {
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (v >= 1e9) return `$${(v / 1e9).toFixed(0)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(0)}M`;
  return `$${v.toLocaleString()}`;
}

interface HeroMetricsProps {
  chartData: ChartPoint[];
  quote: QuoteData | null;
  currentPrice: number | null;
}

export default function HeroMetrics({ chartData, quote, currentPrice }: HeroMetricsProps) {
  const prevClose = quote?.previousClose ?? (chartData.length >= 2 ? chartData[chartData.length - 2]?.close : null);
  const change = prevClose != null && currentPrice != null ? currentPrice - prevClose : 0;
  const changePct = prevClose != null && prevClose > 0 ? (change / prevClose) * 100 : 0;
  const vol = quote?.volume ?? 0;
  const avgVol = quote?.averageVolume ?? 1;
  const volRatio = avgVol > 0 ? vol / avgVol : 0;

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      <div className="bg-dark-card border border-dark-border rounded-lg p-4">
        <p className="text-gray-500 text-sm mb-1">Current Price</p>
        <p className="font-mono text-xl text-accent">
          {currentPrice != null ? `$${currentPrice.toFixed(2)}` : "—"}
        </p>
        {prevClose != null && (
          <p className={`font-mono text-sm ${change >= 0 ? "text-profit" : "text-loss"}`}>
            {change >= 0 ? "+" : ""}{change.toFixed(2)} ({change >= 0 ? "+" : ""}{changePct.toFixed(2)}%)
          </p>
        )}
      </div>
      <div className="bg-dark-card border border-dark-border rounded-lg p-4">
        <p className="text-gray-500 text-sm mb-1">52-Week</p>
        <p className="font-mono text-sm">
          High: {quote?.fiftyTwoWeekHigh != null ? `$${quote.fiftyTwoWeekHigh.toFixed(2)}` : "—"} | Low:{" "}
          {quote?.fiftyTwoWeekLow != null ? `$${quote.fiftyTwoWeekLow.toFixed(2)}` : "—"}
        </p>
      </div>
      <div className="bg-dark-card border border-dark-border rounded-lg p-4">
        <p className="text-gray-500 text-sm mb-1">Market Cap</p>
        <p className="font-mono text-lg text-accent">{quote?.marketCap != null ? fmtMarketCap(quote.marketCap) : "—"}</p>
      </div>
      <div className="bg-dark-card border border-dark-border rounded-lg p-4">
        <p className="text-gray-500 text-sm mb-1">P/E Ratio</p>
        <p className="font-mono text-lg">{quote?.trailingPE != null ? quote.trailingPE.toFixed(2) : "—"}</p>
      </div>
      <div className="bg-dark-card border border-dark-border rounded-lg p-4">
        <p className="text-gray-500 text-sm mb-1">Volume vs Avg</p>
        <p className={`font-mono text-lg ${volRatio > 1.2 ? "text-accent" : volRatio < 0.8 ? "text-gray-400" : ""}`}>
          {avgVol > 0 ? `${volRatio.toFixed(1)}x avg` : "—"}
        </p>
      </div>
      <div className="bg-dark-card border border-dark-border rounded-lg p-4">
        <p className="text-gray-500 text-sm mb-1">Beta</p>
        <p className="font-mono text-lg">{quote?.beta != null ? quote.beta.toFixed(2) : "—"}</p>
      </div>
    </div>
  );
}
