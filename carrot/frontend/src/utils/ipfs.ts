import axios from 'axios';

const PINATA_JWT = import.meta.env.VITE_PINATA_JWT;
const PINATA_API_URL = 'https://api.pinata.cloud/pinning/pinJSONToIPFS';

export interface UploadResult {
  success: boolean;
  ipfsHash?: string;
  ipfsUrl?: string;
  error?: string;
}

export async function uploadToIPFS(data: string | object, filename?: string): Promise<UploadResult> {
  if (!PINATA_JWT) {
    return { success: false, error: 'no pinata jwt bruh. add VITE_PINATA_JWT to .env' };
  }

  try {
    const jsonData = typeof data === 'string' ? { result: data, timestamp: Date.now() } : data;

    const response = await axios.post(PINATA_API_URL, {
      pinataContent: jsonData,
      pinataMetadata: { name: filename || `carrot-result-${Date.now()}.json` },
    }, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${PINATA_JWT}`,
      },
    });

    return {
      success: true,
      ipfsHash: response.data.IpfsHash,
      ipfsUrl: `ipfs://${response.data.IpfsHash}`,
    };
  } catch (error: any) {
    return {
      success: false,
      error: error.response?.data?.error || error.message || 'ipfs upload died',
    };
  }
}

export async function uploadJobResult(jobId: number, logs: string, result: string): Promise<UploadResult> {
  return uploadToIPFS({
    jobId,
    executionLogs: logs,
    result,
    timestamp: new Date().toISOString(),
    platform: 'Carrot GPU DePIN',
  }, `job-${jobId}-result.json`);
}

export function getIPFSGatewayUrl(ipfsHash: string): string {
  let hash = ipfsHash;
  if (hash.startsWith('ipfs://')) {
    hash = hash.slice(7);
  }
  return `https://gateway.pinata.cloud/ipfs/${hash}`;
}
