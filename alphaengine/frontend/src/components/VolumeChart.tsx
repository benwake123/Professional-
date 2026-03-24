import { useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, ReferenceLine, ResponsiveContainer, Tooltip, Cell } from "recharts";
import type { ChartPoint } from "../hooks/useStockData";

type Range = "1M" | "3M" | "6M" | "1Y";
const RANGE_DAYS: Record<Range, number> = { "1M": 21, "3M": 63, "6M": 126, "1Y": 252 };

interface VolumeChartProps {
  chartData: ChartPoint[];
  range: Range;
}

export default function VolumeChart({ chartData, range }: VolumeChartProps) {
  const { data, avgVol } = useMemo(() => {
    const days = Math.min(RANGE_DAYS[range], chartData.length);
    const sliced = chartData.slice(-days);
    const last20 = sliced.slice(-20);
    const total = last20.reduce((s, d) => s + d.volume, 0);
    const avg = last20.length > 0 ? total / last20.length : 0;
    return {
      data: sliced.map((d) => ({
        ...d,
        fill: d.close >= d.open ? "#00ff88" : "#ff4444",
      })),
      avgVol: avg,
    };
  }, [chartData, range]);

  return (
    <div className="bg-dark-card border border-dark-border rounded-lg p-6">
      <h3 className="text-accent font-mono mb-4">Volume</h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <XAxis dataKey="date" stroke="#666" fontSize={10} tickFormatter={(v) => v.slice(5)} />
            <YAxis stroke="#666" fontSize={10} tickFormatter={(v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : `${(v / 1e3).toFixed(0)}K`)} />
            <Tooltip
              contentStyle={{ backgroundColor: "#12121a", border: "1px solid #1e1e2e" }}
              formatter={(v: number) => [v.toLocaleString(), "Volume"]}
              labelFormatter={(l) => l}
            />
            <ReferenceLine y={avgVol} stroke="#666" strokeDasharray="4 4" />
            <Bar dataKey="volume" radius={[2, 2, 0, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.close >= entry.open ? "#00ff88" : "#ff4444"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-gray-500 text-xs mt-2">— 20d avg volume reference</p>
    </div>
  );
}
