export default function SkeletonLoader() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="bg-dark-card border border-dark-border rounded-lg p-4 h-24 bg-gray-800/50" />
        ))}
      </div>
      <div className="bg-dark-card border border-dark-border rounded-lg p-6 h-80 bg-gray-800/50" />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-dark-card border border-dark-border rounded-lg p-6 h-56 bg-gray-800/50" />
        <div className="bg-dark-card border border-dark-border rounded-lg p-6 h-56 bg-gray-800/50" />
      </div>
      <div className="bg-dark-card border border-dark-border rounded-lg p-6 h-64 bg-gray-800/50" />
      <div className="bg-dark-card border border-dark-border rounded-lg p-6 h-24 bg-gray-800/50" />
    </div>
  );
}
