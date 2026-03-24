import { useCallback, useState } from "react";

export interface ChartPoint {
  date: string;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface QuoteData {
  previousClose: number;
  marketCap: number;
  fiftyTwoWeekHigh: number;
  fiftyTwoWeekLow: number;
  volume: number;
  averageVolume: number;
  beta: number | null;
  trailingPE: number | null;
}

export interface StockData {
  ticker: string;
  chartData: ChartPoint[];
  quote: QuoteData | null;
  currentPrice: number | null;
}

interface UseStockDataResult {
  data: StockData | null;
  loading: boolean;
  error: string | null;
  analyze: (ticker: string) => void;
}

const API_BASE = "/api";

async function fetchStockData(ticker: string): Promise<StockData> {
  const res = await fetch(`${API_BASE}/stock/${encodeURIComponent(ticker)}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(res.status === 404 ? text || `Ticker '${ticker}' not found` : text || `HTTP ${res.status}`);
  }
  const json = (await res.json()) as StockData;
  return json;
}

export function useStockData(): UseStockDataResult {
  const [data, setData] = useState<StockData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (ticker: string) => {
    const t = ticker.trim().toUpperCase();
    if (!t) return;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const result = await fetchStockData(t);
      setData(result);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      if (msg === "Failed to fetch" || msg.includes("NetworkError")) {
        setError("Unable to reach server. Start the backend with: cd backend && python -m uvicorn main:app --port 8001");
      } else if (msg.includes("not found") || msg.includes("No")) {
        setError(`Ticker '${t}' not found. Please check the symbol and try again.`);
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  return { data, loading, error, analyze };
}
