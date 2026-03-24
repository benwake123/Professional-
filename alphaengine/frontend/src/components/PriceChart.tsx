import { useState, useMemo } from "react";
import { Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, ComposedChart } from "recharts";
import { movingAverage } from "../utils/calculations";
import type { ChartPoint } from "../hooks/useStockData";

type Range = "1M" | "3M" | "6M" | "1Y";

const RANGE_DAYS: Record<Range, number> = { "1M": 21, "3M": 63, "6M": 126, "1Y": 252 };

interface PriceChartProps {
  chartData: ChartPoint[];
  range: Range;
  onRangeChange: (r: Range) => void;
}

export default function PriceChart({ chartData, range, onRangeChange }: PriceChartProps) {
  const [showMA50, setShowMA50] = useState(true);
  const [showMA200, setShowMA200] = useState(true);

  const { data } = useMemo(() => {
    const days = Math.min(RANGE_DAYS[range], chartData.length);
    const sliced = chartData.slice(-days);
    const closes = sliced.map((d) => d.close);
    const ma50Arr = movingAverage(closes, 50);
    const ma200Arr = movingAverage(closes, 200);
    const combined = sliced.map((d, i) => ({
      ...d,
      ma50: ma50Arr[i],
      ma200: ma200Arr[i],
    }));
    return { data: combined };
  }, [chartData, range]);

  return (
    <div className="bg-dark-card border border-dark-border rounded-lg p-6">
      <div className="flex flex-wrap gap-4 items-center mb-4">
        <div className="flex gap-2">
          {(["1M", "3M", "6M", "1Y"] as Range[]).map((r) => (
            <button
              key={r}
              onClick={() => onRangeChange(r)}
              className={`px-3 py-1 rounded font-mono text-sm ${r === range ? "bg-accent text-dark-bg" : "bg-dark-bg text-gray-400 hover:text-white"}`}
            >
              {r}
            </button>
          ))}
        </div>
        <div className="flex gap-3 items-center text-sm">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={showMA50} onChange={(e) => setShowMA50(e.target.checked)} className="rounded" />
            <span className="text-amber-400">50 MA</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={showMA200} onChange={(e) => setShowMA200(e.target.checked)} className="rounded" />
            <span className="text-purple-400">200 MA</span>
          </label>
        </div>
      </div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data}>
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#00d4ff" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#00d4ff" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="date" stroke="#666" fontSize={11} tickFormatter={(v) => v.slice(5, 7) + "/" + v.slice(2, 4)} />
            <YAxis stroke="#666" fontSize={11} domain={["auto", "auto"]} tickFormatter={(v) => `$${v}`} />
            <Tooltip
              contentStyle={{ backgroundColor: "#12121a", border: "1px solid #1e1e2e" }}
              labelStyle={{ color: "#00d4ff" }}
              formatter={(value: number, name: string) => {
                if (name === "close") return [`$${value.toFixed(2)}`, "Close"];
                if (name === "volume") return [value.toLocaleString(), "Volume"];
                return [value.toFixed(2), name];
              }}
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0]?.payload as { date?: string; close?: number; volume?: number };
                return (
                  <div className="bg-[#12121a] border border-dark-border rounded px-3 py-2">
                    <p className="text-accent font-mono text-sm">{label ?? d?.date}</p>
                    <p className="font-mono text-sm">Close: ${d?.close?.toFixed(2) ?? "—"}</p>
                    <p className="font-mono text-sm text-gray-400">Volume: {d?.volume?.toLocaleString() ?? "—"}</p>
                  </div>
                );
              }}
            />
            <Area type="monotone" dataKey="close" fill="url(#priceGradient)" stroke="none" />
            <Line type="monotone" dataKey="close" stroke="#00d4ff" strokeWidth={2} dot={false} name="close" />
            {showMA50 && <Line type="monotone" dataKey="ma50" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="50 MA" />}
            {showMA200 && <Line type="monotone" dataKey="ma200" stroke="#a855f7" strokeWidth={1.5} dot={false} name="200 MA" />}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
