import { useState } from 'react';
import { useWallet } from './hooks/useWallet';
import ProviderDashboard from './pages/ProviderDashboard';
import ConsumerDashboard from './pages/ConsumerDashboard';
import { getNetworkName } from './config/contracts';

function App() {
  const { address, connectWallet, disconnectWallet, isConnecting, error, network, balance, switchNetwork } = useWallet();
  const [view, setView] = useState<'provider' | 'consumer'>('provider');
  const [showNetworkDropdown, setShowNetworkDropdown] = useState(false);

  return (
    <div className="min-h-screen bg-white text-gray-900">
      <header className="bg-white border-b-2 border-orange-500 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3 cursor-pointer">
                <img src="/carrot-logo.png" alt="Carrot" className="carrot-logo w-10 h-10 object-contain" />
                <h1 className="carrot-title text-3xl">Carrot</h1>
              </div>
              {address && (
                <div className="flex space-x-2">
                  <button
                    onClick={() => setView('provider')}
                    className={`px-4 py-2 rounded-none font-medium transition-all ${
                      view === 'provider'
                        ? 'bg-orange-500 text-white'
                        : 'bg-white text-orange-500 hover:bg-orange-50 border border-orange-500'
                    }`}
                  >
                    Provider
                  </button>
                  <button
                    onClick={() => setView('consumer')}
                    className={`px-4 py-2 rounded-none font-medium transition-all ${
                      view === 'consumer'
                        ? 'bg-orange-500 text-white'
                        : 'bg-white text-orange-500 hover:bg-orange-50 border border-orange-500'
                    }`}
                  >
                    Consumer
                  </button>
                </div>
              )}
            </div>

            <div>
              {!address ? (
                <button
                  onClick={connectWallet}
                  disabled={isConnecting}
                  className="bg-orange-500 hover:bg-orange-600 text-white px-6 py-2 rounded-none font-bold disabled:opacity-50 transition-all"
                >
                  {isConnecting ? 'Connecting...' : 'Connect Freighter'}
                </button>
              ) : (
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <button
                      onClick={() => setShowNetworkDropdown(!showNetworkDropdown)}
                      className="px-4 py-2 rounded-none text-sm font-semibold border-2 border-orange-500 bg-orange-50 text-orange-600"
                    >
                      {getNetworkName(network)} ▼
                    </button>
                    {showNetworkDropdown && (
                      <div className="absolute right-0 mt-2 w-48 bg-white border-2 border-orange-500 rounded-none shadow-lg z-50">
                        <button
                          onClick={() => { switchNetwork('testnet'); setShowNetworkDropdown(false); }}
                          className={`block w-full text-left px-4 py-2 hover:bg-orange-50 ${network === 'testnet' ? 'text-orange-600 bg-orange-50' : 'text-gray-700'}`}
                        >
                          Stellar Testnet
                        </button>
                        <button
                          onClick={() => { switchNetwork('mainnet'); setShowNetworkDropdown(false); }}
                          className={`block w-full text-left px-4 py-2 hover:bg-orange-50 ${network === 'mainnet' ? 'text-orange-600 bg-orange-50' : 'text-gray-700'}`}
                        >
                          Stellar Mainnet
                        </button>
                      </div>
                    )}
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-semibold text-orange-600">
                      {parseFloat(balance).toFixed(2)} XLM
                    </div>
                    <div className="text-xs text-gray-500">
                      {address.slice(0, 8)}...{address.slice(-6)}
                    </div>
                  </div>
                  <button
                    onClick={disconnectWallet}
                    className="bg-white hover:bg-orange-50 text-orange-500 px-4 py-2 rounded-none text-sm border border-orange-500"
                  >
                    Disconnect
                  </button>
                </div>
              )}
            </div>
          </div>
          {error && (
            <div className="mt-4 bg-red-50 border border-red-400 text-red-700 px-4 py-2 rounded-none">
              {error}
            </div>
          )}
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {!address ? (
          <div className="text-center py-20">
            <img src="/carrot-logo.png" alt="Carrot" className="carrot-logo w-32 h-32 object-contain mx-auto mb-8" />
            <h2 className="carrot-title text-5xl mb-4">Welcome to Carrot</h2>
            <p className="text-gray-600 mb-4">GPU Rental on Stellar. Connect your Freighter wallet to start.</p>
            <p className="text-sm text-gray-500">rent gpus, earn xlm, simple as that</p>
          </div>
        ) : (
          <div>
            {view === 'provider' ? (
              <ProviderDashboard address={address} network={network} />
            ) : (
              <ConsumerDashboard address={address} network={network} />
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
