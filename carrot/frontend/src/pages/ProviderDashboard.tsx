import { useState, useEffect } from 'react';
import type { GPU, Job } from '../types';
import { JobStatus, getJobStatusName } from '../types';
import { uploadJobResult, getIPFSGatewayUrl } from '../utils/ipfs';
import {
  type NetworkType,
  stroopsToXLM,
  xlmToStroops,
  registerGPU as registerGPUContract,
  setGPUAvailability,
  getGPU,
  getProviderGPUs,
  claimJob as claimJobContract,
  completeJob as completeJobContract,
  getJob,
  getProviderJobs,
  getNextJobId,
} from '../config/contracts';

interface Props {
  address: string;
  network: NetworkType;
}

export default function ProviderDashboard({ address, network }: Props) {
  const [gpus, setGpus] = useState<(GPU & { id: number })[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [availableJobs, setAvailableJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(false);
  const [txPending, setTxPending] = useState(false);
  const [notifications, setNotifications] = useState<string[]>([]);

  const [gpuModel, setGpuModel] = useState('');
  const [vram, setVram] = useState('');
  const [price, setPrice] = useState('');

  const [processingJobId, setProcessingJobId] = useState<number | null>(null);
  const [executionLogs, setExecutionLogs] = useState<Record<number, string>>({});
  const [executionResults, setExecutionResults] = useState<Record<number, string>>({});
  const [uploadingToIPFS, setUploadingToIPFS] = useState<Record<number, boolean>>({});
  const [ipfsHashes, setIpfsHashes] = useState<Record<number, string>>({});

  const addNotification = (msg: string) => {
    setNotifications(prev => [msg, ...prev].slice(0, 5));
    setTimeout(() => setNotifications(prev => prev.slice(0, -1)), 8000);
  };

  const loadData = async () => {
    setLoading(true);
    try {
      const gpuIds = await getProviderGPUs(network, address, address);
      const gpuList: (GPU & { id: number })[] = [];

      if (gpuIds && Array.isArray(gpuIds)) {
        for (const id of gpuIds) {
          try {
            const gpu = await getGPU(network, address, Number(id));
            if (gpu) {
              gpuList.push({
                id: Number(id),
                provider: gpu.provider,
                model: gpu.model,
                vramGB: Number(gpu.vram_gb),
                pricePerHour: stroopsToXLM(gpu.price_per_hour),
                available: gpu.available,
                totalJobs: Number(gpu.total_jobs),
                registeredAt: Number(gpu.registered_at),
              });
            }
          } catch (e) {
            console.log(`failed to load gpu ${id}:`, e);
          }
        }
      }
      setGpus(gpuList.filter(g => g.available));

      const jobIds = await getProviderJobs(network, address, address);
      const jobList: Job[] = [];

      if (jobIds && Array.isArray(jobIds)) {
        for (const id of jobIds) {
          try {
            const job = await getJob(network, address, Number(id));
            if (job) {
              jobList.push({
                jobId: Number(job.job_id),
                consumer: job.consumer,
                gpuId: Number(job.gpu_id),
                description: job.description,
                computeHours: Number(job.compute_hours),
                paymentAmount: stroopsToXLM(job.payment_amount),
                provider: job.provider,
                status: job.status,
                createdAt: Number(job.created_at),
                claimedAt: Number(job.claimed_at),
                completedAt: Number(job.completed_at),
                resultHash: job.result_hash,
              });
            }
          } catch (e) {
            console.log(`failed to load job ${id}:`, e);
          }
        }
      }
      setJobs(jobList);

      const myGpuIds = gpuList.map(g => g.id);
      const nextJobId = await getNextJobId(network, address);
      const openJobs: Job[] = [];

      for (let i = 0; i < Number(nextJobId); i++) {
        try {
          const job = await getJob(network, address, i);
          if (job && job.status === 0 && myGpuIds.includes(Number(job.gpu_id))) {
            openJobs.push({
              jobId: Number(job.job_id),
              consumer: job.consumer,
              gpuId: Number(job.gpu_id),
              description: job.description,
              computeHours: Number(job.compute_hours),
              paymentAmount: stroopsToXLM(job.payment_amount),
              provider: job.provider,
              status: job.status,
              createdAt: Number(job.created_at),
              claimedAt: Number(job.claimed_at),
              completedAt: Number(job.completed_at),
              resultHash: job.result_hash,
            });
          }
        } catch (e) {}
      }
      setAvailableJobs(openJobs);

    } catch (e) {
      console.error('load failed:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadData(); }, [address, network]);

  const registerGPU = async () => {
    if (!gpuModel || !vram || !price) return;
    setTxPending(true);
    try {
      const priceStroops = xlmToStroops(parseFloat(price));
      await registerGPUContract(network, address, gpuModel, parseInt(vram), priceStroops);
      addNotification(`registered ${gpuModel} - lfg`);
      setGpuModel(''); setVram(''); setPrice('');
      await loadData();
    } catch (e: any) {
      alert(`failed: ${e.message}`);
    } finally {
      setTxPending(false);
    }
  };

  const toggleAvailability = async (gpuId: number, currentStatus: boolean) => {
    setTxPending(true);
    try {
      await setGPUAvailability(network, address, gpuId, !currentStatus);
      addNotification(`gpu #${gpuId} ${!currentStatus ? 'online' : 'offline'}`);
      await loadData();
    } catch (e: any) {
      alert(`failed: ${e.message}`);
    } finally {
      setTxPending(false);
    }
  };

  const claimJob = async (jobId: number) => {
    setTxPending(true);
    try {
      await claimJobContract(network, address, jobId);
      addNotification(`claimed job #${jobId} - get to work`);
      await loadData();
    } catch (e: any) {
      alert(`failed: ${e.message}`);
    } finally {
      setTxPending(false);
    }
  };

  const parseJobData = (desc: string) => {
    try { return JSON.parse(desc); } catch { return { type: 'simple', description: desc }; }
  };

  const runJobWithGPU = async (job: Job) => {
    const jobData = parseJobData(job.description);
    setProcessingJobId(job.jobId);
    setExecutionLogs(prev => ({ ...prev, [job.jobId]: 'spinning up gpu worker...\n' }));

    try {
      const health = await fetch('http://localhost:3001/health').catch(() => null);
      if (!health) throw new Error('gpu worker not running. cd provider-worker && npm start');

      setExecutionLogs(prev => ({ ...prev, [job.jobId]: prev[job.jobId] + 'sending job to worker...\n' }));

      const res = await fetch('http://localhost:3001/process-job', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jobId: job.jobId,
          jobType: jobData.type || 'simple',
          jobData: jobData.type === 'python-script' ? { code: jobData.code }
            : jobData.type === 'docker-image' ? { image: jobData.image }
            : { description: jobData.description }
        })
      });

      const result = await res.json();

      if (result.success) {
        setExecutionLogs(prev => ({ ...prev, [job.jobId]: prev[job.jobId] + '\n=== done ===\n' + result.logs }));
        const resultValue = result.resultHash || result.result;
        setExecutionResults(prev => ({ ...prev, [job.jobId]: resultValue }));
        if (resultValue && (resultValue.startsWith('ipfs://') || resultValue.startsWith('Qm'))) {
          setIpfsHashes(prev => ({ ...prev, [job.jobId]: resultValue }));
          addNotification(`job #${job.jobId} finished - image uploaded to IPFS`);
        } else {
          addNotification(`job #${job.jobId} finished`);
        }
      } else {
        throw new Error(result.error || 'gpu go brrrr failed');
      }
    } catch (e: any) {
      setExecutionLogs(prev => ({ ...prev, [job.jobId]: prev[job.jobId] + `\nERROR: ${e.message}` }));
      alert(`gpu error: ${e.message}`);
    } finally {
      setProcessingJobId(null);
    }
  };

  const uploadResultToIPFS = async (jobId: number) => {
    setUploadingToIPFS(prev => ({ ...prev, [jobId]: true }));
    try {
      const result = await uploadJobResult(jobId, executionLogs[jobId] || '', executionResults[jobId] || '');
      if (result.success && result.ipfsUrl) {
        setIpfsHashes(prev => ({ ...prev, [jobId]: result.ipfsUrl! }));
        setExecutionResults(prev => ({ ...prev, [jobId]: result.ipfsUrl! }));
        addNotification(`uploaded to ipfs: ${result.ipfsHash}`);
      } else {
        throw new Error(result.error || 'ipfs died');
      }
    } catch (e: any) {
      alert(`ipfs fail: ${e.message}`);
    } finally {
      setUploadingToIPFS(prev => ({ ...prev, [jobId]: false }));
    }
  };

  const completeJobAction = async (jobId: number) => {
    let resultHash: string = ipfsHashes[jobId] || executionResults[jobId] || '';
    if (!resultHash) {
      const userInput = prompt('enter result hash (ipfs://... or whatever):');
      if (!userInput) return;
      resultHash = userInput;
    }

    setTxPending(true);
    try {
      await completeJobContract(network, address, jobId, resultHash);
      addNotification(`job #${jobId} completed - payment incoming`);
      setExecutionLogs(prev => { const n = { ...prev }; delete n[jobId]; return n; });
      setExecutionResults(prev => { const n = { ...prev }; delete n[jobId]; return n; });
      setIpfsHashes(prev => { const n = { ...prev }; delete n[jobId]; return n; });
      await loadData();
    } catch (e: any) {
      alert(`failed: ${e.message}`);
    } finally {
      setTxPending(false);
    }
  };

  return (
    <div className="space-y-8">
      {notifications.length > 0 && (
        <div className="fixed top-20 right-4 space-y-2 z-50 max-w-md">
          {notifications.map((n, i) => (
            <div key={i} className="bg-orange-500 text-white px-6 py-3 rounded-none shadow-lg border-2 border-orange-400">{n}</div>
          ))}
        </div>
      )}

      <div className="flex items-center space-x-4 mb-8">
        <img src="/carrot-logo.png" alt="Carrot" className="carrot-logo w-12 h-12 object-contain" />
        <h2 className="carrot-title text-3xl">Provider Dashboard</h2>
      </div>

      {!loading && gpus.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-orange-50 p-4 rounded-none border-2 border-orange-500">
            <div className="text-xs text-gray-500 mb-1">Total GPUs</div>
            <div className="text-2xl font-bold text-orange-500">{gpus.length}</div>
            <div className="text-xs text-gray-600 mt-1">{gpus.filter(g => g.available).length} online</div>
          </div>
          <div className="bg-orange-50 p-4 rounded-none border-2 border-orange-500">
            <div className="text-xs text-gray-500 mb-1">Total Jobs</div>
            <div className="text-2xl font-bold text-orange-500">{gpus.reduce((s, g) => s + g.totalJobs, 0)}</div>
          </div>
          <div className="bg-orange-50 p-4 rounded-none border-2 border-orange-500">
            <div className="text-xs text-gray-500 mb-1">Total Earned</div>
            <div className="text-2xl font-bold text-orange-500">
              {jobs.filter(j => j.status === JobStatus.Completed).reduce((s, j) => s + parseFloat(j.paymentAmount) * 0.95, 0).toFixed(2)} XLM
            </div>
          </div>
          <div className="bg-orange-50 p-4 rounded-none border-2 border-orange-500">
            <div className="text-xs text-gray-500 mb-1">Active Jobs</div>
            <div className="text-2xl font-bold text-orange-500">{jobs.filter(j => j.status === JobStatus.Claimed).length}</div>
          </div>
        </div>
      )}

      <div className="bg-gray-50 p-6 rounded-none border border-gray-200">
        <h3 className="text-xl font-semibold mb-4 text-orange-500">Register New GPU</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <input type="text" placeholder="GPU Model (RTX 4090)" value={gpuModel} onChange={e => setGpuModel(e.target.value)}
            className="bg-white border-2 border-gray-300 rounded-none px-4 py-2 text-gray-900 focus:border-orange-500 focus:outline-none" />
          <input type="number" placeholder="VRAM (GB)" value={vram} onChange={e => setVram(e.target.value)}
            className="bg-white border-2 border-gray-300 rounded-none px-4 py-2 text-gray-900 focus:border-orange-500 focus:outline-none" />
          <input type="text" placeholder="Price per Hour (XLM)" value={price} onChange={e => setPrice(e.target.value)}
            className="bg-white border-2 border-gray-300 rounded-none px-4 py-2 text-gray-900 focus:border-orange-500 focus:outline-none" />
          <button onClick={registerGPU} disabled={txPending || !gpuModel || !vram || !price}
            className="bg-orange-500 hover:bg-orange-600 text-white px-6 py-2 rounded-none font-medium disabled:opacity-50 transition-all">
            {txPending ? 'Processing...' : 'Register GPU'}
          </button>
        </div>
      </div>

      <div>
        <h3 className="text-2xl font-semibold mb-4 text-orange-500">My GPUs</h3>
        {loading ? <div className="text-gray-600">loading from stellar...</div> : gpus.length === 0 ? <div className="text-gray-600">no gpus registered yet</div> : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {gpus.map(gpu => (
              <div key={gpu.id} className="bg-white p-4 rounded-none border-2 border-gray-200 hover:border-orange-500 transition-all shadow-sm">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="text-lg font-semibold text-orange-500">{gpu.model}</h4>
                  <span className={`px-2 py-1 rounded-none text-xs text-white ${gpu.available ? 'bg-orange-500' : 'bg-gray-400'}`}>
                    {gpu.available ? 'Online' : 'Offline'}
                  </span>
                </div>
                <div className="text-sm space-y-1 text-gray-700">
                  <div>VRAM: {gpu.vramGB} GB</div>
                  <div>Price: {parseFloat(gpu.pricePerHour).toFixed(2)} XLM/hour</div>
                  <div>Total Jobs: {gpu.totalJobs}</div>
                </div>
                <div className="grid grid-cols-2 gap-2 mt-4">
                  <button onClick={() => toggleAvailability(gpu.id, gpu.available)} disabled={txPending}
                    className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-2 rounded-none text-sm disabled:opacity-50 border border-gray-300">
                    {gpu.available ? 'Go Offline' : 'Go Online'}
                  </button>
                  <button onClick={() => toggleAvailability(gpu.id, true)} disabled={txPending}
                    className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-none text-sm disabled:opacity-50">
                    Remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div>
        <h3 className="text-2xl font-semibold mb-4 text-orange-500">Available Jobs</h3>
        {loading ? <div className="text-gray-600">loading...</div> : availableJobs.length === 0 ?
          <div className="text-gray-600">no jobs waiting for your gpus</div> : (
          <div className="space-y-4">
            {availableJobs.map(job => {
              const gpu = gpus.find(g => g.id === job.gpuId);
              const earnings = (parseFloat(job.paymentAmount) * 0.95).toFixed(2);
              return (
                <div key={job.jobId} className="bg-orange-50 p-6 rounded-none border-2 border-orange-500">
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h4 className="text-xl font-semibold text-orange-500">Job #{job.jobId}</h4>
                        <span className="px-3 py-1 rounded-none text-sm font-medium bg-orange-500 text-white">OPEN</span>
                      </div>
                      <p className="text-gray-700 text-lg mb-3">{parseJobData(job.description).description || job.description}</p>
                      <div className="grid grid-cols-3 gap-4 mt-3 p-3 bg-white rounded-none border border-orange-300">
                        <div>
                          <div className="text-xs text-gray-500">Total Payment</div>
                          <div className="text-lg font-semibold text-gray-900">{parseFloat(job.paymentAmount).toFixed(2)} XLM</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-500">You Earn (95%)</div>
                          <div className="text-lg font-semibold text-orange-500">{earnings} XLM</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-500">GPU</div>
                          <div className="text-sm text-orange-500">{gpu?.model || `#${job.gpuId}`}</div>
                        </div>
                      </div>
                    </div>
                    <button onClick={() => claimJob(job.jobId)} disabled={txPending}
                      className="ml-4 bg-orange-500 hover:bg-orange-600 text-white px-8 py-4 rounded-none font-bold text-lg disabled:opacity-50 shadow-lg">
                      Claim Job<br/><span className="text-sm font-normal">Earn {earnings} XLM</span>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div>
        <h3 className="text-2xl font-semibold mb-4 text-orange-500">Current Jobs</h3>
        {jobs.filter(j => j.status === JobStatus.Claimed).length === 0 ?
          <div className="text-gray-600">no active jobs</div> : (
          <div className="space-y-4">
            {jobs.filter(j => j.status === JobStatus.Claimed).map(job => {
              const jobData = parseJobData(job.description);
              const earnings = (parseFloat(job.paymentAmount) * 0.95).toFixed(2);
              return (
                <div key={job.jobId} className="bg-white p-6 rounded-none border-2 border-gray-200 shadow-sm">
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h4 className="text-xl font-semibold text-orange-500">Job #{job.jobId}</h4>
                        <span className="px-3 py-1 rounded-none text-xs bg-gray-200 text-gray-700">{jobData.type || 'simple'}</span>
                        <span className="px-3 py-1 rounded-none text-sm font-medium bg-orange-500 text-white">{getJobStatusName(job.status)}</span>
                      </div>
                      <p className="text-gray-700 text-lg mb-3">{jobData.description || job.description}</p>

                      {jobData.type === 'python-script' && jobData.code && (
                        <div className="mb-3 p-3 bg-gray-100 rounded-none border border-gray-300">
                          <div className="text-xs text-gray-500 mb-1">Python Code:</div>
                          <pre className="text-xs text-gray-800 font-mono overflow-x-auto max-h-32">{jobData.code.substring(0, 200)}{jobData.code.length > 200 && '...'}</pre>
                        </div>
                      )}

                      <div className="grid grid-cols-2 gap-4 mt-3 p-3 bg-gray-50 rounded-none border border-gray-200">
                        <div><div className="text-xs text-gray-500">Total</div><div className="text-lg font-semibold text-gray-900">{parseFloat(job.paymentAmount).toFixed(2)} XLM</div></div>
                        <div><div className="text-xs text-gray-500">Your Cut (95%)</div><div className="text-lg font-semibold text-orange-500">{earnings} XLM</div></div>
                      </div>

                      {executionLogs[job.jobId] && (
                        <div className="mt-4 p-4 bg-gray-100 rounded-none border border-gray-300">
                          <div className="text-sm text-gray-600 mb-2">Execution Logs:</div>
                          <pre className="text-xs text-gray-800 font-mono overflow-x-auto max-h-64 whitespace-pre-wrap">{executionLogs[job.jobId]}</pre>
                        </div>
                      )}

                      {executionResults[job.jobId] && !ipfsHashes[job.jobId] && (
                        <div className="mt-3 p-3 bg-orange-50 rounded-none border border-orange-300">
                          <div className="text-xs text-gray-500 mb-1">Result:</div>
                          <div className="text-sm text-gray-800 font-mono break-all">{executionResults[job.jobId]}</div>
                        </div>
                      )}

                      {ipfsHashes[job.jobId] && (
                        <div className="mt-3 p-3 bg-orange-50 rounded-none border-2 border-orange-500">
                          <div className="text-xs text-gray-500 mb-1">IPFS Result:</div>
                          <div className="text-sm text-gray-800 font-mono break-all mb-2">{ipfsHashes[job.jobId]}</div>
                          <a href={getIPFSGatewayUrl(ipfsHashes[job.jobId])} target="_blank" rel="noopener noreferrer"
                            className="bg-orange-500 hover:bg-orange-600 text-white px-4 py-2 rounded-none text-sm font-semibold inline-block">
                            View Result
                          </a>
                        </div>
                      )}
                    </div>

                    <div className="ml-4 flex flex-col gap-2">
                      {(jobData.type === 'python-script' || jobData.type === 'docker-image') && (
                        <button onClick={() => runJobWithGPU(job)} disabled={processingJobId === job.jobId || txPending}
                          className="bg-orange-500 hover:bg-orange-600 text-white px-6 py-3 rounded-none font-medium disabled:opacity-50 whitespace-nowrap">
                          {processingJobId === job.jobId ? 'Processing...' : 'Run with GPU'}
                        </button>
                      )}
                      {executionResults[job.jobId] && !ipfsHashes[job.jobId] && (
                        <button onClick={() => uploadResultToIPFS(job.jobId)} disabled={uploadingToIPFS[job.jobId]}
                          className="bg-gray-500 hover:bg-gray-600 text-white px-6 py-3 rounded-none font-medium disabled:opacity-50 whitespace-nowrap">
                          {uploadingToIPFS[job.jobId] ? 'Uploading...' : 'Upload to IPFS'}
                        </button>
                      )}
                      <button onClick={() => completeJobAction(job.jobId)} disabled={txPending}
                        className="bg-orange-500 hover:bg-orange-600 text-white px-6 py-3 rounded-none font-medium disabled:opacity-50 whitespace-nowrap">
                        {ipfsHashes[job.jobId] ? 'Submit to Chain' : 'Complete Job'}<br/><span className="text-xs">Earn {earnings} XLM</span>
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="mt-16 pt-8 border-t border-gray-200 flex items-center justify-center space-x-3 opacity-70">
        <img src="/carrot-logo.png" alt="Carrot" className="carrot-logo w-6 h-6 object-contain" />
        <span className="text-sm text-gray-500">Powered by <span className="carrot-title">Carrot</span> on Stellar</span>
      </div>
    </div>
  );
}
