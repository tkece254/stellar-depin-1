import { useState, useEffect, useCallback } from 'react';
import { isConnected, getAddress, signTransaction, getNetwork as getFreighterNetwork, isAllowed, setAllowed } from '@stellar/freighter-api';
import { Horizon } from '@stellar/stellar-sdk';
import { getNetwork, DEFAULT_NETWORK, type NetworkType } from '../config/contracts';

export const useWallet = () => {
  const [address, setAddress] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [network, setNetwork] = useState<NetworkType>(DEFAULT_NETWORK);
  const [balance, setBalance] = useState<string>('0');

  const fetchBalance = useCallback(async (publicKey: string) => {
    try {
      const networkConfig = getNetwork(network);
      const server = new Horizon.Server(networkConfig.horizonUrl);
      const account = await server.loadAccount(publicKey);
      const xlmBalance = account.balances.find((b: any) => b.asset_type === 'native');
      setBalance(xlmBalance ? xlmBalance.balance : '0');
    } catch (err: any) {
      if (err?.response?.status === 404) {
        setBalance('0');
      } else {
        console.log('balance fetch failed:', err);
      }
    }
  }, [network]);

  const connectWallet = async () => {
    setIsConnecting(true);
    setError(null);

    const getErrorMessage = (err: any): string => {
      if (!err) return 'Unknown error';
      if (typeof err === 'string') return err;
      if (err.message) return err.message;
      return JSON.stringify(err);
    };

    try {
      // check if freighter is installed
      const connectedResult = await isConnected();
      if (connectedResult.error) {
        throw new Error(getErrorMessage(connectedResult.error));
      }
      if (!connectedResult.isConnected) {
        throw new Error('Freighter not installed. Get it at freighter.app');
      }

      // check if we already have permission
      const allowedResult = await isAllowed();
      if (allowedResult.error) {
        throw new Error(getErrorMessage(allowedResult.error));
      }

      // request access if not already allowed
      if (!allowedResult.isAllowed) {
        const accessResult = await setAllowed();
        if (accessResult.error) {
          throw new Error(getErrorMessage(accessResult.error));
        }
        if (!accessResult.isAllowed) {
          throw new Error('User denied access. Click Connect again and approve in Freighter.');
        }
      }

      // get the address
      const addressResult = await getAddress();
      if (addressResult.error) {
        throw new Error(getErrorMessage(addressResult.error));
      }
      if (!addressResult.address) {
        throw new Error('Could not get address. Unlock Freighter and try again.');
      }

      setAddress(addressResult.address);
      await fetchBalance(addressResult.address);

      // sync network from freighter
      const freighterNet = await getFreighterNetwork();
      if (!freighterNet.error) {
        if (freighterNet.network === 'TESTNET') {
          setNetwork('testnet');
        } else if (freighterNet.network === 'PUBLIC') {
          setNetwork('mainnet');
        }
      }
    } catch (err: any) {
      const msg = err.message || 'wallet connection failed';
      setError(msg);
      console.error('wallet error:', msg);
    } finally {
      setIsConnecting(false);
    }
  };

  const disconnectWallet = () => {
    setAddress(null);
    setBalance('0');
    setError(null);
  };

  const switchNetwork = (newNetwork: NetworkType) => {
    setNetwork(newNetwork);
    if (address) fetchBalance(address);
  };

  useEffect(() => {
    if (!address) return;
    fetchBalance(address);
    const interval = setInterval(() => fetchBalance(address), 15000);
    return () => clearInterval(interval);
  }, [address, fetchBalance]);

  return {
    address,
    isConnecting,
    error,
    network,
    balance,
    connectWallet,
    disconnectWallet,
    switchNetwork,
    signTransaction,
  };
};
