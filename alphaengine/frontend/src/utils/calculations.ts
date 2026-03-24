/** Pure calculation functions for stock analysis. All math per spec. */

/** Daily return: (close[t] - close[t-1]) / close[t-1] */
export function dailyReturns(closes: number[]): number[] {
  const ret: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    ret.push((closes[i]! - closes[i - 1]!) / closes[i - 1]!);
  }
  return ret;
}

/** Annualized vol: std(daily_returns) * sqrt(252) */
export function annualizedVolatility(returns: number[]): number {
  if (returns.length < 2) return 0;
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((s, r) => s + (r - mean) ** 2, 0) / (returns.length - 1);
  return Math.sqrt(variance) * Math.sqrt(252);
}

/** Sharpe: (mean(daily_returns) * 252 - 0.05) / (std(daily_returns) * sqrt(252)) */
export function sharpeRatio(returns: number[], riskFreeRate = 0.05): number {
  if (returns.length < 2) return 0;
  const annRet = (1 + returns.reduce((a, b) => a + b, 0) / returns.length) ** 252 - 1;
  const annVol = annualizedVolatility(returns);
  return annVol > 0 ? (annRet - riskFreeRate) / annVol : 0;
}

/** Sortino: (annRet - rf) / (std(negative_returns) * sqrt(252)); downside deviation */
export function sortinoRatio(returns: number[], riskFreeRate = 0.05): number {
  const neg = returns.filter((r) => r < 0);
  if (neg.length < 2) return returns.length > 0 ? 999 : 0;
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const annRet = (1 + mean) ** 252 - 1;
  const ddMean = neg.reduce((a, b) => a + b, 0) / neg.length;
  const ddVar = neg.reduce((s, r) => s + (r - ddMean) ** 2, 0) / (neg.length - 1);
  const downsideVol = Math.sqrt(ddVar) * Math.sqrt(252);
  return downsideVol > 0 ? (annRet - riskFreeRate) / downsideVol : 0;
}

/** Max drawdown: max((peak - trough) / peak) over rolling cumulative max. Returns [maxDD, troughIndex]. */
export function maxDrawdown(closes: number[]): { maxDrawdown: number; troughDateIndex: number } {
  let peak = closes[0] ?? 0;
  let maxDD = 0;
  let troughIdx = 0;
  for (let i = 1; i < closes.length; i++) {
    const c = closes[i] ?? 0;
    if (c > peak) peak = c;
    const dd = peak > 0 ? (peak - c) / peak : 0;
    if (dd > maxDD) {
      maxDD = dd;
      troughIdx = i;
    }
  }
  return { maxDrawdown: maxDD, troughDateIndex: troughIdx };
}

/** RSI: 14-period, Wilder smoothing (exponential with alpha=1/14) */
export function rsi(closes: number[], period = 14): number {
  if (closes.length < period + 1) return 50;
  let avgGain = 0;
  let avgLoss = 0;
  for (let i = 1; i <= period; i++) {
    const ch = (closes[i] ?? 0) - (closes[i - 1] ?? 0);
    if (ch > 0) avgGain += ch;
    else avgLoss += Math.abs(ch);
  }
  avgGain /= period;
  avgLoss /= period;
  for (let i = period + 1; i < closes.length; i++) {
    const ch = (closes[i] ?? 0) - (closes[i - 1] ?? 0);
    avgGain = (avgGain * (period - 1) + (ch > 0 ? ch : 0)) / period;
    avgLoss = (avgLoss * (period - 1) + (ch < 0 ? Math.abs(ch) : 0)) / period;
  }
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

/** VaR 95%: 5th percentile of daily_returns (historical) */
export function varHistorical(returns: number[], percentile: number): number {
  if (returns.length === 0) return 0;
  const sorted = [...returns].sort((a, b) => a - b);
  const idx = Math.max(0, Math.floor((percentile / 100) * returns.length));
  return sorted[idx] ?? 0;
}

/** Moving average */
export function movingAverage(values: number[], period: number): (number | null)[] {
  const result: (number | null)[] = [];
  for (let i = 0; i < values.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else {
      let sum = 0;
      for (let j = 0; j < period; j++) {
        sum += values[i - j] ?? 0;
      }
      result.push(sum / period);
    }
  }
  return result;
}

/** Skewness of returns */
export function skewness(returns: number[]): number {
  if (returns.length < 3) return 0;
  const n = returns.length;
  const mean = returns.reduce((a, b) => a + b, 0) / n;
  const m2 = returns.reduce((s, r) => s + (r - mean) ** 2, 0) / n;
  const m3 = returns.reduce((s, r) => s + (r - mean) ** 3, 0) / n;
  const std = Math.sqrt(m2);
  return std > 0 ? m3 / std ** 3 : 0;
}

/** Kurtosis (excess) */
export function kurtosis(returns: number[]): number {
  if (returns.length < 4) return 0;
  const n = returns.length;
  const mean = returns.reduce((a, b) => a + b, 0) / n;
  const m2 = returns.reduce((s, r) => s + (r - mean) ** 2, 0) / n;
  const m4 = returns.reduce((s, r) => s + (r - mean) ** 4, 0) / n;
  return m2 > 0 ? m4 / m2 ** 2 - 3 : 0;
}

/** Calmar: annualized return / abs(max drawdown) */
export function calmarRatio(closes: number[]): number {
  const rets = dailyReturns(closes);
  if (rets.length < 2) return 0;
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const annRet = (1 + mean) ** 252 - 1;
  const { maxDrawdown: mdd } = maxDrawdown(closes);
  return mdd > 0 ? annRet / mdd : 0;
}

/** Total return over period */
export function totalReturn(closes: number[]): number {
  if (closes.length < 2) return 0;
  return (closes[closes.length - 1]! - closes[0]!) / closes[0]!;
}
