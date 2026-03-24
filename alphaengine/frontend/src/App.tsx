import { useState } from "react";
import { useStockData } from "./hooks/useStockData";
import AnalysisDashboard from "./components/AnalysisDashboard";
import SkeletonLoader from "./components/SkeletonLoader";

const SUGGESTED_TICKERS = ["AAPL", "NVDA", "TSLA", "SPY", "BTC-USD"];

export default function App() {
  const [input, setInput] = useState("");
  const { data, loading, error, analyze } = useStockData();

  const handleAnalyze = () => {
    if (input.trim()) analyze(input.trim());
  };

  const handleSuggestedClick = (ticker: string) => {
    setInput(ticker);
    analyze(ticker);
  };

  return (
    <div className="min-h-screen bg-dark-bg text-gray-200">
      <header className="sticky top-0 z-10 bg-dark-bg/95 border-b border-dark-border backdrop-blur">
        <div className="max-w-6xl mx-auto px-4 py-4 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="font-mono font-semibold text-accent text-xl">α AlphaEngine</div>
          <div className="flex-1 w-full md:max-w-md flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value.toUpperCase())}
              onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}
              placeholder="Enter ticker (e.g. AAPL)"
              className="flex-1 bg-dark-card border border-dark-border rounded-lg px-4 py-2 font-mono text-accent placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-accent"
            />
            <button
              onClick={handleAnalyze}
              disabled={loading || !input.trim()}
              className="px-6 py-2 bg-accent text-dark-bg font-mono rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "..." : "Analyze"}
            </button>
          </div>
          {data && (
            <span className="px-3 py-1 bg-accent/20 text-accent font-mono rounded-full text-sm">
              {data.ticker}
            </span>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        {!data && !loading && !error && (
          <div className="flex flex-col items-center justify-center min-h-[60vh] gap-8">
            <div className="flex flex-col items-center gap-4 w-full max-w-md">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value.toUpperCase())}
                onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}
                placeholder="Enter ticker symbol"
                className="w-full bg-dark-card border border-dark-border rounded-lg px-6 py-4 font-mono text-xl text-accent placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-accent text-center"
              />
              <button
                onClick={handleAnalyze}
                disabled={loading || !input.trim()}
                className="w-full px-6 py-3 bg-accent text-dark-bg font-mono text-lg rounded-lg hover:opacity-90 disabled:opacity-50"
              >
                Analyze
              </button>
            </div>
            <p className="text-gray-500 text-sm">Suggested:</p>
            <div className="flex flex-wrap justify-center gap-2">
              {SUGGESTED_TICKERS.map((t) => (
                <button
                  key={t}
                  onClick={() => handleSuggestedClick(t)}
                  className="px-4 py-2 bg-dark-card border border-dark-border rounded-full font-mono text-sm text-gray-400 hover:text-accent hover:border-accent/50 transition"
                >
                  {t}
                </button>
              ))}
            </div>
          </div>
        )}

        {loading && <SkeletonLoader />}

        {error && (
          <div className="flex flex-col items-center justify-center min-h-[40vh] gap-4">
            <p className="text-loss font-mono text-center">{error}</p>
            <button
              onClick={() => handleAnalyze()}
              className="px-6 py-2 bg-dark-card border border-dark-border rounded-lg font-mono hover:border-accent/50"
            >
              Try Again
            </button>
          </div>
        )}

        {data && !loading && <AnalysisDashboard data={data} />}
      </main>
    </div>
  );
}
