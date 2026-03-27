import { Contract, Networks, TransactionBuilder, rpc, xdr, scValToNative, nativeToScVal, Address } from '@stellar/stellar-sdk';
import { signTransaction } from '@stellar/freighter-api';

export const NETWORKS = {
  testnet: {
    name: 'Stellar Testnet',
    networkPassphrase: Networks.TESTNET,
    horizonUrl: 'https://horizon-testnet.stellar.org',
    sorobanUrl: 'https://soroban-testnet.stellar.org',
    gpuRegistry: 'CAHUSLBSYQETDXGUEDPMA6QNLRJBX2JW7BYL2LXSWYBLZN543FKM5LAX',
    jobMarketplace: 'CAWNKF4EFHJYJPHZSIXYPG5D45HFGLBNFN6NGO3676YTRLNNXGKJ7PMW',
    nativeAsset: 'XLM',
  },
  mainnet: {
    name: 'Stellar Mainnet',
    networkPassphrase: Networks.PUBLIC,
    horizonUrl: 'https://horizon.stellar.org',
    sorobanUrl: 'https://soroban.stellar.org',
    gpuRegistry: 'DEPLOY_ME',
    jobMarketplace: 'DEPLOY_ME',
    nativeAsset: 'XLM',
  },
};

export type NetworkType = 'testnet' | 'mainnet';
export const DEFAULT_NETWORK: NetworkType = 'testnet';

export function getNetwork(network: NetworkType = DEFAULT_NETWORK) {
  return NETWORKS[network];
}

export function getNetworkName(network: NetworkType): string {
  return NETWORKS[network].name;
}

export function stroopsToXLM(stroops: bigint | number | string): string {
  return (Number(stroops) / 10_000_000).toFixed(7);
}

export function xlmToStroops(xlm: number): bigint {
  return BigInt(Math.floor(xlm * 10_000_000));
}

export async function callContract(
  network: NetworkType,
  contractId: string,
  method: string,
  args: xdr.ScVal[],
  publicKey: string,
  isReadOnly: boolean = false
): Promise<any> {
  const config = getNetwork(network);
  const server = new rpc.Server(config.sorobanUrl);
  const contract = new Contract(contractId);
  const sourceAccount = await server.getAccount(publicKey);

  let tx = new TransactionBuilder(sourceAccount, {
    fee: '100000',
    networkPassphrase: config.networkPassphrase,
  })
    .addOperation(contract.call(method, ...args))
    .setTimeout(60)
    .build();

  const simulated = await server.simulateTransaction(tx);

  if (rpc.Api.isSimulationError(simulated)) {
    throw new Error(`Simulation failed: ${simulated.error}`);
  }

  if (isReadOnly) {
    if (simulated.result) {
      return scValToNative(simulated.result.retval);
    }
    return null;
  }

  tx = rpc.assembleTransaction(tx, simulated).build();
  const xdrTx = tx.toXDR();

  const signedResult = await signTransaction(xdrTx, {
    networkPassphrase: config.networkPassphrase,
  });

  if (signedResult.error) {
    throw new Error(`Signing error: ${signedResult.error}`);
  }

  if (!signedResult.signedTxXdr) {
    throw new Error('Transaction signing cancelled');
  }

  const signedTx = TransactionBuilder.fromXDR(signedResult.signedTxXdr, config.networkPassphrase);
  const sendResponse = await server.sendTransaction(signedTx);

  if (sendResponse.status === 'ERROR') {
    throw new Error(`Transaction failed: ${sendResponse.errorResult}`);
  }

  let getResponse = await server.getTransaction(sendResponse.hash);
  while (getResponse.status === 'NOT_FOUND') {
    await new Promise(r => setTimeout(r, 1000));
    getResponse = await server.getTransaction(sendResponse.hash);
  }

  if (getResponse.status === 'SUCCESS' && getResponse.returnValue) {
    return scValToNative(getResponse.returnValue);
  }

  if (getResponse.status === 'FAILED') {
    throw new Error('Transaction failed on chain');
  }

  return null;
}

// gpu registry helpers
export async function registerGPU(network: NetworkType, publicKey: string, model: string, vramGB: number, pricePerHour: bigint) {
  const config = getNetwork(network);
  return callContract(network, config.gpuRegistry, 'register_gpu', [
    new Address(publicKey).toScVal(),
    nativeToScVal(model, { type: 'string' }),
    nativeToScVal(vramGB, { type: 'u32' }),
    nativeToScVal(pricePerHour, { type: 'i128' }),
  ], publicKey, false);
}

export async function setGPUAvailability(network: NetworkType, publicKey: string, gpuId: number, available: boolean) {
  const config = getNetwork(network);
  return callContract(network, config.gpuRegistry, 'set_availability', [
    new Address(publicKey).toScVal(),
    nativeToScVal(gpuId, { type: 'u32' }),
    nativeToScVal(available, { type: 'bool' }),
  ], publicKey, false);
}

export async function getGPU(network: NetworkType, publicKey: string, gpuId: number) {
  const config = getNetwork(network);
  return callContract(network, config.gpuRegistry, 'get_gpu', [
    nativeToScVal(gpuId, { type: 'u32' }),
  ], publicKey, true);
}

export async function getProviderGPUs(network: NetworkType, publicKey: string, provider: string) {
  const config = getNetwork(network);
  return callContract(network, config.gpuRegistry, 'get_provider_gpus', [
    new Address(provider).toScVal(),
  ], publicKey, true);
}

export async function getNextGPUId(network: NetworkType, publicKey: string) {
  const config = getNetwork(network);
  return callContract(network, config.gpuRegistry, 'get_next_gpu_id', [], publicKey, true);
}

// job marketplace helpers
export async function postJob(network: NetworkType, publicKey: string, gpuId: number, description: string, computeHours: number, paymentAmount: bigint) {
  const config = getNetwork(network);
  return callContract(network, config.jobMarketplace, 'post_job', [
    new Address(publicKey).toScVal(),
    nativeToScVal(gpuId, { type: 'u32' }),
    nativeToScVal(description, { type: 'string' }),
    nativeToScVal(computeHours, { type: 'u32' }),
    nativeToScVal(paymentAmount, { type: 'i128' }),
  ], publicKey, false);
}

export async function claimJob(network: NetworkType, publicKey: string, jobId: number) {
  const config = getNetwork(network);
  return callContract(network, config.jobMarketplace, 'claim_job', [
    new Address(publicKey).toScVal(),
    nativeToScVal(jobId, { type: 'u32' }),
  ], publicKey, false);
}

export async function completeJob(network: NetworkType, publicKey: string, jobId: number, resultHash: string) {
  const config = getNetwork(network);
  return callContract(network, config.jobMarketplace, 'complete_job', [
    new Address(publicKey).toScVal(),
    nativeToScVal(jobId, { type: 'u32' }),
    nativeToScVal(resultHash, { type: 'string' }),
  ], publicKey, false);
}

export async function cancelJob(network: NetworkType, publicKey: string, jobId: number) {
  const config = getNetwork(network);
  return callContract(network, config.jobMarketplace, 'cancel_job', [
    new Address(publicKey).toScVal(),
    nativeToScVal(jobId, { type: 'u32' }),
  ], publicKey, false);
}

export async function getJob(network: NetworkType, publicKey: string, jobId: number) {
  const config = getNetwork(network);
  return callContract(network, config.jobMarketplace, 'get_job', [
    nativeToScVal(jobId, { type: 'u32' }),
  ], publicKey, true);
}

export async function getConsumerJobs(network: NetworkType, publicKey: string, consumer: string) {
  const config = getNetwork(network);
  return callContract(network, config.jobMarketplace, 'get_consumer_jobs', [
    new Address(consumer).toScVal(),
  ], publicKey, true);
}

export async function getProviderJobs(network: NetworkType, publicKey: string, provider: string) {
  const config = getNetwork(network);
  return callContract(network, config.jobMarketplace, 'get_provider_jobs', [
    new Address(provider).toScVal(),
  ], publicKey, true);
}

export async function getNextJobId(network: NetworkType, publicKey: string) {
  const config = getNetwork(network);
  return callContract(network, config.jobMarketplace, 'get_next_job_id', [], publicKey, true);
}
