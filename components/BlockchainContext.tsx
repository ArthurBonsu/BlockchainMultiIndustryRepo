import React, { createContext, useContext, useState, PropsWithChildren, useEffect, useCallback, useMemo } from 'react';

import Web3 from 'web3';
import { Contract } from 'web3-eth-contract';
import { loadContracts, PasschainContract } from '../utils/contract-loader';
import { Logger } from '../utils/logger';
import { useTransactions } from '../contexts/TransactionContext';
import networkConfig from '../config/network_config';
import { createTatumBlockchainService, TatumConnectionOptions } from '../services/tatum-blockchain-service';
import { TatumConstants } from '../config/tatum-config';
import { BlockchainService } from '../services/blockchain-service';
import { CrossChainService, CrossChainTransactionData } from '../services/cross-chain-service';
// Existing type definitions

// Comprehensive Connection Options
interface ConnectionOptions {
  method?: 'metamask' | 'tatum';
  network?: string;
  tatumOptions?: TatumConnectionOptions;
}

// Enhanced Blockchain Context Type
interface BlockchainContextType {
  web3: Web3 | null;
  accounts: string[];
  contracts: {
    [key: string]: ContractType;
  };
  connect: (options?: ConnectionOptions) => Promise<{
    success: boolean;
    accounts?: string[];
    error?: string;
  }>;
  disconnect: () => void;
  processTransaction: (data: any) => Promise<TransactionResult>;
  switchNetwork?: (options: NetworkSwitchOptions) => Promise<void>;
  blockchainService?: BlockchainService | null;
  isLoading: boolean;
  error: string | null;
  networkInfo: {
    chainId?: number;
    name?: string;
    type?: 'mainnet' | 'testnet';
  } | null;
}



// Updated type definition
type ContractType = Contract<any> & {
  options?: { 
    address?: string 
  };
} | (PasschainContract & {
  options?: { 
    address?: string 
  };
});
interface NetworkSwitchOptions {
  chainId?: number;
  networkName?: string;
}

interface TransactionResult {
  parsedMetadata: string;
  speculativeTx: string;
  zkProof: string;
  clusterProcessing: string;
  relayCrossChain: string;
  crossChainHashes?: string[];
}

const BlockchainContext = createContext<BlockchainContextType>({
  web3: null,
  accounts: [],
  contracts: {},
  connect: async () => ({ success: false }),
  disconnect: () => {},
  processTransaction: async () => {
    throw new Error('BlockchainContext not initialized');
  },
  isLoading: false,
  error: null,
  networkInfo: null
});

export const BlockchainProvider: React.FC<PropsWithChildren> = ({ children }) => {
  const [web3, setWeb3] = useState<Web3 | null>(null);
  const [accounts, setAccounts] = useState<string[]>([]);
  const [contracts, setContracts] = useState<{[key: string]: ContractType}>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [networkInfo, setNetworkInfo] = useState<{
    chainId?: number;
    name?: string;
    type?: 'mainnet' | 'testnet';
  } | null>(null);
  
  const [blockchainService, setBlockchainService] = useState<BlockchainService | null>(null);
  const [isClient, setIsClient] = useState(false);
  const [isContractInitialized, setIsContractInitialized] = useState(false);

  // Use transactions context
  const { addTransaction } = useTransactions();

  // Type-safe contract address retrieval
  const getContractAddress = (contract: ContractType): string => {
    if ('address' in contract && contract.address) {
      return contract.address;
    }
    
    if ('options' in contract && contract.options?.address) {
      return contract.options.address;
    }
    
    return 'Address not found';
  };

  const connectMetaMask = async (networkName?: string): Promise<{
    success: boolean;
    accounts?: string[];
    error?: string;
  }> => {
    try {
      // Check for Ethereum provider
      const ethereum = (window as any).ethereum;
      if (!ethereum) {
        throw new Error('Ethereum wallet not found. Please install MetaMask.');
      }

      // Request account access
      const accs = await ethereum.request({ method: 'eth_requestAccounts' });

      // Create Web3 instance
      const web3Instance = new Web3(ethereum);

      // Optional network switching
      if (networkName) {
        const networkDetails = networkConfig.networks[networkName];
        if (networkDetails) {
          await ethereum.request({
            method: 'wallet_switchEthereumChain',
            params: [{ chainId: `0x${networkDetails.chainId.toString(16)}` }]
          });
        }
      }

      // Load contracts
      const loadedContracts = await loadBlockchainContracts(web3Instance);
      
      // Create blockchain service
      const service = new BlockchainService(web3Instance, loadedContracts);

      // Update state
      setWeb3(web3Instance);
      setAccounts(accs);
      setBlockchainService(service);
      setNetworkInfo({
        chainId: Number(await web3Instance.eth.getChainId().toString()), // Explicit conversion
        name: networkName || 'Unknown',
        type: 'testnet'
      });

      Logger.info('MetaMask connected successfully', {
        network: networkName,
        accounts: accs
      });

      return {
        success: true,
        accounts: accs
      };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'MetaMask connection failed';
      
      Logger.error('MetaMask connection error', { 
        error: errorMessage, 
        details: err 
      });

      setError(errorMessage);
      
      return {
        success: false,
        error: errorMessage
      };
    }
  };

  const connectTatum = async (options: TatumConnectionOptions = {}): Promise<{
    success: boolean;
    accounts?: string[];
    error?: string;
  }> => {
    setIsLoading(true);
    setError(null);

    try {
      // Use provided options or defaults
      const connectionOptions = {
        apiKey: options.apiKey || TatumConstants.API_KEY,
        network: options.network || TatumConstants.DEFAULT_NETWORK
      };

      // Create Tatum service
      const tatumService = createTatumBlockchainService(connectionOptions);

      // Test connection
      const connectionTest = await tatumService.testConnection();
      
      if (!connectionTest.connected) {
        throw new Error('Failed to establish Tatum blockchain connection');
      }

      // Get Web3 instance
      const web3Instance = tatumService['web3'];
      
      // Load contracts
      const loadedContracts = await loadBlockchainContracts(web3Instance);
      
      // Create blockchain service
      const service = new BlockchainService(web3Instance, loadedContracts);

      // Update state
      setWeb3(web3Instance);
      setAccounts(connectionTest.accounts || []);
      setNetworkInfo({
        chainId: connectionTest.networkInfo?.chainId,
        name: connectionTest.networkInfo?.name,
        type: connectionTest.networkInfo?.type
      });
      setBlockchainService(service);

      Logger.info('Tatum connected successfully', {
        network: connectionOptions.network,
        accounts: connectionTest.accounts
      });

      return {
        success: true,
        accounts: connectionTest.accounts
      };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Tatum connection failed';
      
      Logger.error('Tatum connection error', { 
        error: errorMessage, 
        details: err 
      });

      setError(errorMessage);
      
      return {
        success: false,
        error: errorMessage
      };
    } finally {
      setIsLoading(false);
    }
  };

 // Wrap connect in useCallback
 const connect = useCallback(async (options: ConnectionOptions = {}): Promise<{
  success: boolean;
  accounts?: string[];
  error?: string;
}> => {
  const method = options.method || 'metamask';

  switch (method) {
    case 'metamask':
      return await connectMetaMask(options.network);
    case 'tatum':
      return await connectTatum(options.tatumOptions);
    default:
      return {
        success: false,
        error: 'Invalid connection method'
      };
  }
}, []); //

  const disconnect = () => {
    setWeb3(null);
    setAccounts([]);
    setContracts({});
    setBlockchainService(null);
    setNetworkInfo(null);
    setError(null);
    setIsContractInitialized(false);
    Logger.info('Blockchain connection disconnected');
  };

  const loadBlockchainContracts = async (web3Instance: Web3): Promise<{ [key: string]: PasschainContract }> => {
    try {
      setIsLoading(true);
      const loadedContracts = await loadContracts(web3Instance);
      
      // Verify each contract was loaded correctly
      const verifiedContracts: { [key: string]: PasschainContract } = {};
      for (const [name, contract] of Object.entries(loadedContracts)) {
        if (!contract || !contract.methods) {
          Logger.warn(`Contract ${name} failed to load correctly`);
          continue;
        }
        verifiedContracts[name] = contract;
      }
  
      if (Object.keys(verifiedContracts).length === 0) {
        throw new Error('No contracts could be loaded');
      }
      
      setContracts(verifiedContracts);
      setIsContractInitialized(true);
      Logger.info('Contracts loaded successfully', { 
        contracts: Object.keys(verifiedContracts),
        contractAddresses: Object.fromEntries(
          Object.entries(verifiedContracts).map(([name, contract]) => [
            name, 
            contract.address || contract.options.address
          ])
        )
      });
  
      return verifiedContracts;
    } catch (error) {
      Logger.error('Failed to load contracts', error);
      setIsContractInitialized(false);
      throw new Error('Contract loading failed');
    } finally {
      setIsLoading(false);
    }
  };

  const switchNetwork = async (options: NetworkSwitchOptions): Promise<void> => {
    if (!blockchainService) {
      throw new Error('Blockchain service not initialized');
    }
    
    try {
      await blockchainService.switchNetwork(options);
      
      // Optional: Reconnect or refresh contracts after network switch
      if (web3) {
        await loadBlockchainContracts(web3);
      }
    } catch (error) {
      Logger.error('Network switch failed', error);
      throw error;
    }
  };

  const processTransaction = useCallback(async (data: any): Promise<TransactionResult> => {
    // Check if it's client-side
    if (!isClient) {
      throw new Error('Transaction processing is client-side only');
    }
  
    // Check if blockchain is still loading
    if (isLoading) {
      throw new Error('Blockchain is still initializing');
    }
  
    const startTime = performance.now();
  
    try {
      // Ensure connection and contracts
      if (!web3 || accounts.length === 0) {
        const connectResult = await connect();
        if (!connectResult.success) {
          throw new Error('Failed to establish blockchain connection');
        }
      }
  
      // Initialize contracts if not already done
      if (!isContractInitialized) {
        await loadBlockchainContracts(web3!);
      }
  
      // Validate required contracts are present
      const requiredContracts = [
        'MetadataParser', 
        'PacechainChannel', 
        'TransactionValidator', 
        'ZKPVerifierBase', 
        'ClusterManager', 
        'TransactionRelay'
      ];
  
      const missingContracts = requiredContracts.filter(
        contractName => !contracts[contractName]
      );
  
      if (missingContracts.length > 0) {
        throw new Error(`Missing required contracts: ${missingContracts.join(', ')}`);
      }
  
      // Destructure required contracts
      const { 
        MetadataParser, 
        PacechainChannel, 
        ZKPVerifierBase, 
        ClusterManager, 
        TransactionRelay 
      } = contracts;
  
      // Prepare transaction data
      const encodedData = web3!.utils.fromAscii(JSON.stringify(data));
      const defaultOptions = { from: accounts[0] };
  
      // Execute blockchain transactions
      const parsedMetadata = await MetadataParser.methods
        .parseMetadata(encodedData)
        .send(defaultOptions);
  
      const speculativeTx = await PacechainChannel.methods
        .initiateSpeculativeTransaction(encodedData)
        .send(defaultOptions);
  
      const zkProof = await ZKPVerifierBase.methods
        .generateProof(encodedData, web3!.utils.fromAscii('witness'))
        .send(defaultOptions);
  
      const clusterProcessing = await ClusterManager.methods
        .processTransaction(encodedData)
        .send(defaultOptions);
  
      const relayCrossChain = await TransactionRelay.methods
        .relayTransaction(encodedData)
        .send(defaultOptions);
  
      // Prepare result object
      const result: TransactionResult = {
        parsedMetadata: parsedMetadata.transactionHash,
        speculativeTx: speculativeTx.transactionHash,
        zkProof: zkProof.transactionHash,
        clusterProcessing: clusterProcessing.transactionHash,
        relayCrossChain: relayCrossChain.transactionHash
      };
  
      // Calculate processing time
      const endTime = performance.now();
      const processingTime = endTime - startTime;
  
      // Prepare cross-chain transaction data
      const crossChainData: CrossChainTransactionData = {
        ...data,
        ethereumAddress: accounts[0],
        polkadotAddress: data.polkadotAddress 
      };

      // Process cross-chain transaction
      const crossChainResult = await CrossChainService.processCrossChainTransaction(crossChainData);
  
      // Merge local and cross-chain results
      const mergedResult = {
        ...result,
        crossChainHashes: crossChainResult.transactionHashes
      };
  
      // Add transaction to context
      addTransaction({
        ...data,
        blockchainResults: mergedResult,
        processingTime
      });
  
      Logger.info('Transaction processed successfully', mergedResult);
      return mergedResult;
    } catch (error) {
      Logger.error('Transaction processing error', error);
      throw error;
    }
  }, [
    // Dependencies for useCallback
    isClient, 
    web3, 
    accounts, 
    contracts, 
    connect, 
    loadBlockchainContracts,
    addTransaction
  ]);

  // Client-side initialization effect
  useEffect(() => {
    setIsClient(true);
  }, []);

  // Auto-connect effect
  useEffect(() => {
    const autoConnect = async () => {
      if (isClient) {
        // Prioritize Tatum connection if API key is available
        if (TatumConstants.API_KEY) {
          await connect({ method: 'tatum' });
        } else {
          // Fallback to MetaMask
          await connect({ method: 'metamask' });
        }
      }
    };

    autoConnect();
  }, [isClient, connect]); // Add connect to dependency array

  // Render provider
  return (
    <BlockchainContext.Provider value={{ 
      web3, 
      accounts, 
      contracts, 
      connect,
      disconnect, 
      processTransaction,
      switchNetwork,
      blockchainService,
      isLoading,
      error,
      networkInfo
    }}>
      {children}
    </BlockchainContext.Provider>
  );
};
export const useBlockchain = () => useContext(BlockchainContext);

// Helper function to check blockchain connection possibility
export const canConnectToBlockchain = () => 
  !!(TatumConstants.API_KEY || (window as any).ethereum);