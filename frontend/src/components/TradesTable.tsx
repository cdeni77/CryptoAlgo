import { Trade } from '../types';

interface TradesTableProps {
  trades: Trade[];
  loading?: boolean;
}

export default function TradesTable({ trades, loading = false }: TradesTableProps) {
  if (loading) {
    return (
      <div className="glass rounded-2xl p-12 flex flex-col items-center justify-center h-96">
        <div className="w-16 h-16 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mb-6"></div>
        <p className="text-gray-400">Loading trade history...</p>
      </div>
    );
  }

  return (
    <div className="glass rounded-2xl overflow-hidden shadow-2xl">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-indigo-900/50">
          <thead className="bg-gray-900/70">
            <tr>
              <th className="px-6 py-5 text-left text-sm font-semibold text-indigo-300">ID</th>
              <th className="px-6 py-5 text-left text-sm font-semibold text-indigo-300">Coin</th>
              <th className="px-6 py-5 text-left text-sm font-semibold text-indigo-300">Open Time</th>
              <th className="px-6 py-5 text-left text-sm font-semibold text-indigo-300">Side</th>
              <th className="px-6 py-5 text-left text-sm font-semibold text-indigo-300">Contracts</th>
              <th className="px-6 py-5 text-left text-sm font-semibold text-indigo-300">Entry</th>
              <th className="px-6 py-5 text-left text-sm font-semibold text-indigo-300">Exit</th>
              <th className="px-6 py-5 text-left text-sm font-semibold text-indigo-300">PNL</th>
              <th className="px-6 py-5 text-left text-sm font-semibold text-indigo-300">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {trades.length === 0 ? (
              <tr>
                <td colSpan={9} className="px-6 py-16 text-center text-gray-400">
                  <svg className="w-16 h-16 mx-auto mb-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-xl font-medium">No trades recorded yet</p>
                  <p className="mt-2 text-gray-500">Your bot will automatically populate this table</p>
                </td>
              </tr>
            ) : (
              trades.map((trade) => (
                <tr key={trade.id} className="hover:bg-indigo-950/20 transition-colors">
                  <td className="px-6 py-5 text-sm text-gray-300">{trade.id}</td>
                  <td className="px-6 py-5 text-sm font-medium text-white">{trade.coin}</td>
                  <td className="px-6 py-5 text-sm text-gray-400">
                    {new Date(trade.datetime_open).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}
                  </td>
                  <td className="px-6 py-5">
                    <span className={`inline-flex px-4 py-1.5 rounded-full text-xs font-semibold tracking-wide ${
                      trade.side === 'long' 
                        ? 'bg-green-900/70 text-green-300 border border-green-700/50' 
                        : 'bg-red-900/70 text-red-300 border border-red-700/50'
                    }`}>
                      {trade.side.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-6 py-5 text-sm text-gray-300">{trade.contracts.toFixed(4)}</td>
                  <td className="px-6 py-5 text-sm text-gray-300">${trade.entry_price.toLocaleString()}</td>
                  <td className="px-6 py-5 text-sm text-gray-300">
                    {trade.exit_price ? `$${trade.exit_price.toLocaleString()}` : '—'}
                  </td>
                  <td className={`px-6 py-5 text-sm font-semibold ${
                    trade.net_pnl === null ? 'text-gray-500' :
                    trade.net_pnl > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {trade.net_pnl !== null ? `$${trade.net_pnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}` : '—'}
                  </td>
                  <td className="px-6 py-5">
                    <span className={`inline-flex px-4 py-1.5 rounded-full text-xs font-semibold tracking-wide ${
                      trade.status === 'open' 
                        ? 'bg-yellow-900/70 text-yellow-300 border border-yellow-700/50' 
                        : 'bg-gray-800 text-gray-300 border border-gray-700'
                    }`}>
                      {trade.status.toUpperCase()}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}