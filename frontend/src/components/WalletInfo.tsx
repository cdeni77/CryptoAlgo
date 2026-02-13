import { useEffect, useState } from 'react';
import { getWallet } from '../api/walletApi';

interface WalletData {
  balance: number;
  realized_pnl: number;
  unrealized_pnl: number;
  total_pnl: number;
  coinbase?: {
    spot?: { value_usd: number | null; status: string };
    perps?: { value_usd: number | null; status: string };
    total_value_usd: number | null;
  };
}

interface WalletInfoProps {
  loading: boolean;
}

export default function WalletInfo({ loading }: WalletInfoProps) {
  const [wallet, setWallet] = useState<WalletData | null>(null);

  useEffect(() => {
    const loadWallet = async () => {
      try {
        const data = await getWallet();
        setWallet(data);
      } catch (err) {
        console.error('Wallet fetch error:', err);
      }
    };

    loadWallet();
    const interval = setInterval(loadWallet, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading || !wallet) {
    return (
      <div className="glass-card rounded-xl p-10 flex items-center justify-center h-40">
        <div className="w-10 h-10 border-2 border-[var(--accent-cyan)] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const fmt = (v: number) =>
    `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  const pnlColor = (v: number) =>
    v >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]';

  const items = [
    { label: 'Balance', value: fmt(wallet.balance), color: 'text-[var(--text-primary)]' },
    { label: 'Realized PNL', value: `${wallet.realized_pnl >= 0 ? '+' : ''}${fmt(wallet.realized_pnl)}`, color: pnlColor(wallet.realized_pnl) },
    { label: 'Unrealized PNL', value: `${wallet.unrealized_pnl >= 0 ? '+' : ''}${fmt(wallet.unrealized_pnl)}`, color: pnlColor(wallet.unrealized_pnl) },
    { label: 'Total PNL', value: `${wallet.total_pnl >= 0 ? '+' : ''}${fmt(wallet.total_pnl)}`, color: pnlColor(wallet.total_pnl) },
    {
      label: 'Coinbase Spot',
      value: wallet.coinbase?.spot?.value_usd != null ? fmt(wallet.coinbase.spot.value_usd) : 'N/A',
      color: 'text-[var(--text-primary)]',
    },
    {
      label: 'Coinbase Perps',
      value: wallet.coinbase?.perps?.value_usd != null ? fmt(wallet.coinbase.perps.value_usd) : 'N/A',
      color: 'text-[var(--text-primary)]',
    },
    {
      label: 'Coinbase Total',
      value: wallet.coinbase?.total_value_usd != null ? fmt(wallet.coinbase.total_value_usd) : 'N/A',
      color: 'text-[var(--accent-cyan)]',
    },
  ];

  return (
    <div className="glass-card rounded-xl p-5">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {items.map(({ label, value, color }) => (
          <div key={label} className="p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-subtle)]">
            <p className="text-[10px] font-mono-trade text-[var(--text-muted)] uppercase tracking-wider mb-1.5">{label}</p>
            <p className={`text-xl font-bold font-mono-trade ${color}`}>{value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
