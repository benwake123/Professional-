import { useState } from "react";
import HeroMetrics from "./HeroMetrics";
import PriceChart from "./PriceChart";
import VolumeChart from "./VolumeChart";
import ReturnsDistribution from "./ReturnsDistribution";
import QuantMetricsPanel from "./QuantMetricsPanel";
import MomentumSignals from "./MomentumSignals";
import type { StockData } from "../hooks/useStockData";

type Range = "1M" | "3M" | "6M" | "1Y";

interface AnalysisDashboardProps {
  data: StockData;
}

export default function AnalysisDashboard({ data }: AnalysisDashboardProps) {
  const [range, setRange] = useState<Range>("1Y");
  return (
    <div className="space-y-6">
      <HeroMetrics chartData={data.chartData} quote={data.quote} currentPrice={data.currentPrice} />
      <PriceChart chartData={data.chartData} range={range} onRangeChange={setRange} />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <VolumeChart chartData={data.chartData} range={range} />
        <ReturnsDistribution chartData={data.chartData} range={range} />
      </div>
      <QuantMetricsPanel chartData={data.chartData} />
      <MomentumSignals chartData={data.chartData} />
    </div>
  );
}
