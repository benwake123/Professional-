import { useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, ReferenceLine, ResponsiveContainer, Tooltip, Cell } from "recharts";
import { dailyReturns } from "../utils/calculations";
import type { ChartPoint } from "../hooks/useStockData";

type Range = "1M" | "3M" | "6M" | "1Y";
const RANGE_DAYS: Record<Range, number> = { "1M": 21, "3M": 63, "6M": 126, "1Y": 252 };

interface ReturnsDistributionProps {
  chartData: ChartPoint[];
  range: Range;
}

const BINS = 20;

export default function ReturnsDistribution({ chartData, range }: ReturnsDistributionProps) {
  const { bins, mean, std } = useMemo(() => {
    const days = Math.min(RANGE_DAYS[range], chartData.length);
    const closes = chartData.slice(-days).map((d) => d.close);
    const rets = dailyReturns(closes);
    if (rets.length === 0) return { bins: [], mean: 0, std: 0 };
    const meanRet = rets.reduce((a, b) => a + b, 0) / rets.length;
    const variance = rets.reduce((s, r) => s + (r - meanRet) ** 2, 0) / rets.length;
    const stdRet = Math.sqrt(variance);
    const min = Math.min(...rets);
    const max = Math.max(...rets);
    const step = max > min ? (max - min) / BINS : 0.0001;
    const counts: { binCenter: number; count: number; fill: string }[] = [];
    for (let i = 0; i < BINS; i++) {
      const lo = min + i * step;
      const hi = min + (i + 1) * step;
      const center = (lo + hi) / 2;
      const cnt = rets.filter((r) => r >= lo && (i === BINS - 1 ? r <= hi : r < hi)).length;
      counts.push({
        binCenter: center * 100,
        count: cnt,
        fill: center >= 0 ? "#00ff88" : "#ff4444",
      });
    }
    return { bins: counts, mean: meanRet * 100, std: stdRet * 100 };
  }, [chartData, range]);

  return (
    <div className="bg-dark-card border border-dark-border rounded-lg p-6">
      <h3 className="text-accent font-mono mb-4">Returns Distribution</h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins} margin={{ top: 4, right: 4, left: 4, bottom: 4 }}>
            <XAxis dataKey="binCenter" stroke="#666" fontSize={10} tickFormatter={(v) => `${Number(v).toFixed(2)}%`} />
            <YAxis stroke="#666" fontSize={10} />
            <Tooltip
              contentStyle={{ backgroundColor: "#12121a", border: "1px solid #1e1e2e" }}
              formatter={(v: number) => [v, "Count"]}
              labelFormatter={(l) => `Return: ${Number(l).toFixed(2)}%`}
            />
            <ReferenceLine x={0} stroke="#666" strokeWidth={1} />
            <Bar dataKey="count" radius={[2, 2, 0, 0]}>
              {bins.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-gray-500 text-xs mt-2">
        μ = {mean.toFixed(3)}% | σ = {std.toFixed(3)}%
      </p>
    </div>
  );
}
