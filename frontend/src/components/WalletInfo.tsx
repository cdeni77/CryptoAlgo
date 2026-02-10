import { useEffect, useState } from 'react';
import { getWallet } from '../api/walletApi';

interface WalletData {
  balance: number;
  realized_pnl: number;
  unrealized_pnl: number;
  total_pnl: number;
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
    const interval = setInterval(loadWallet, 5000);  // Refresh every 5s
    return () => clearInterval(interval);
  }, []);

  if (loading || !wallet) {
    return (
      <div className="glass rounded-2xl p-12 flex items-center justify-center h-48">
        <div className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  const formatCurrency = (value: number) => 
    `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  const pnlClass = (value: number) => 
    value >= 0 ? 'text-green-400' : 'text-red-400';

  return (
    <div className="glass rounded-2xl p-8 shadow-2xl border border-indigo-900/30">
      <h3 className="text-2xl font-bold mb-6 bg-gradient-to-r from-blue-300 to-indigo-400 bg-clip-text text-transparent">
        Wallet Overview
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="p-6 bg-gray-900/50 rounded-xl">
          <h4 className="text-sm text-gray-400 mb-2">Balance</h4>
          <p className="text-2xl font-bold text-white">{formatCurrency(wallet.balance)}</p>
        </div>
        
        <div className="p-6 bg-gray-900/50 rounded-xl">
          <h4 className="text-sm text-gray-400 mb-2">Realized PNL</h4>
          <p className={`text-2xl font-bold ${pnlClass(wallet.realized_pnl)}`}>
            {wallet.realized_pnl >= 0 ? '+' : ''}{formatCurrency(wallet.realized_pnl)}
          </p>
        </div>
        
        <div className="p-6 bg-gray-900/50 rounded-xl">
          <h4 className="text-sm text-gray-400 mb-2">Unrealized PNL</h4>
          <p className={`text-2xl font-bold ${pnlClass(wallet.unrealized_pnl)}`}>
            {wallet.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(wallet.unrealized_pnl)}
          </p>
        </div>
        
        <div className="p-6 bg-gray-900/50 rounded-xl">
          <h4 className="text-sm text-gray-400 mb-2">Total PNL</h4>
          <p className={`text-2xl font-bold ${pnlClass(wallet.total_pnl)}`}>
            {wallet.total_pnl >= 0 ? '+' : ''}{formatCurrency(wallet.total_pnl)}
          </p>
        </div>
      </div>
    </div>
  );
}