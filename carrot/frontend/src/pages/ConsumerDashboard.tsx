import { useState, useEffect } from 'react';
import type { GPU, Job } from '../types';
import { JobStatus, getJobStatusName } from '../types';
import { getIPFSGatewayUrl } from '../utils/ipfs';
import {
  type NetworkType,
  stroopsToXLM,
  xlmToStroops,
  getGPU,
  getNextGPUId,
  postJob as postJobContract,
  cancelJob as cancelJobContract,
  getJob,
  getConsumerJobs,
} from '../config/contracts';

interface Props {
  address: string;
  network: NetworkType;
}

export default function ConsumerDashboard({ address, network }: Props) {
  const [availableGPUs, setAvailableGPUs] = useState<(GPU & { id: number })[]>([]);
  const [myJobs, setMyJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(false);
  const [txPending, setTxPending] = useState(false);
  const [notifications, setNotifications] = useState<string[]>([]);

  const [selectedGpuId, setSelectedGpuId] = useState<number | null>(null);
  const [jobDescription, setJobDescription] = useState('');
  const [computeHours, setComputeHours] = useState('');
  const [jobType, setJobType] = useState<'simple' | 'python-script' | 'docker-image'>('simple');
  const [pythonCode, setPythonCode] = useState('');
  const [dockerImage, setDockerImage] = useState('');

  const [searchTerm, setSearchTerm] = useState('');
  const [minVram, setMinVram] = useState('');
  const [maxPrice, setMaxPrice] = useState('');
  const [sortBy, setSortBy] = useState<'price' | 'vram' | 'jobs'>('price');

  const selectedGpu = availableGPUs.find(g => g.id === selectedGpuId);
  const calculatedPayment = selectedGpu && computeHours
    ? (parseFloat(selectedGpu.pricePerHour) * parseFloat(computeHours)).toFixed(2)
    : '0';

  const filteredGPUs = availableGPUs
    .filter(gpu => {
      const matchesSearch = gpu.model.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesVram = !minVram || gpu.vramGB >= parseInt(minVram);
      const matchesPrice = !maxPrice || parseFloat(gpu.pricePerHour) <= parseFloat(maxPrice);
      return matchesSearch && matchesVram && matchesPrice;
    })
    .sort((a, b) => {
      if (sortBy === 'price') return parseFloat(a.pricePerHour) - parseFloat(b.pricePerHour);
      if (sortBy === 'vram') return b.vramGB - a.vramGB;
      return b.totalJobs - a.totalJobs;
    });

  const addNotification = (msg: string) => {
    setNotifications(prev => [msg, ...prev].slice(0, 5));
    setTimeout(() => setNotifications(prev => prev.slice(0, -1)), 8000);
  };

  const loadData = async () => {
    setLoading(true);
    try {
      // load all available gpus
      const nextGpuId = await getNextGPUId(network, address);
      const gpuList: (GPU & { id: number })[] = [];

      for (let i = 0; i < Number(nextGpuId); i++) {
        try {
          const gpu = await getGPU(network, address, i);
          if (gpu && gpu.available) {
            gpuList.push({
              id: i,
              provider: gpu.provider,
              model: gpu.model,
              vramGB: Number(gpu.vram_gb),
              pricePerHour: stroopsToXLM(gpu.price_per_hour),
              available: gpu.available,
              totalJobs: Number(gpu.total_jobs),
              registeredAt: Number(gpu.registered_at),
            });
          }
        } catch (e) {}
      }
      setAvailableGPUs(gpuList);

      // load consumer's jobs
      const jobIds = await getConsumerJobs(network, address, address);
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
          } catch (e) {}
        }
      }
      setMyJobs(jobList);

    } catch (e) {
      console.error('load failed:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadData(); }, [address, network]);

  const postJob = async () => {
    if (selectedGpuId === null || !computeHours || calculatedPayment === '0') return;

    let fullDescription = jobDescription;
    if (jobType === 'python-script') {
      if (!pythonCode.trim()) { alert('enter some code'); return; }
      fullDescription = JSON.stringify({ type: 'python-script', description: jobDescription, code: pythonCode });
    } else if (jobType === 'docker-image') {
      if (!dockerImage.trim()) { alert('enter docker image'); return; }
      fullDescription = JSON.stringify({ type: 'docker-image', description: jobDescription, image: dockerImage });
    } else {
      if (!jobDescription.trim()) { alert('enter job description'); return; }
      fullDescription = JSON.stringify({ type: 'simple', description: jobDescription });
    }

    setTxPending(true);
    try {
      const paymentStroops = xlmToStroops(parseFloat(calculatedPayment));
      await postJobContract(network, address, selectedGpuId, fullDescription, parseInt(computeHours), paymentStroops);
      addNotification(`job posted - ${calculatedPayment} XLM locked in escrow`);
      setSelectedGpuId(null); setJobDescription(''); setComputeHours('');
      setPythonCode(''); setDockerImage(''); setJobType('simple');
      await loadData();
    } catch (e: any) {
      alert(`failed: ${e.message}`);
    } finally {
      setTxPending(false);
    }
  };

  const cancelJob = async (jobId: number) => {
    if (!confirm('cancel job? payment will be refunded')) return;
    setTxPending(true);
    try {
      await cancelJobContract(network, address, jobId);
      addNotification(`job #${jobId} cancelled - xlm refunded`);
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

  return (
    <div className="space-y-8">
      {notifications.length > 0 && (
        <div className="fixed top-20 right-4 space-y-2 z-50 max-w-md">
          {notifications.map((n, i) => (
            <div key={i} className="bg-orange-600 text-gray-900 px-6 py-3 rounded-none shadow-lg border-2 border-orange-400 animate-pulse">{n}</div>
          ))}
        </div>
      )}

      <div className="flex items-center space-x-4 mb-8">
        <img src="/carrot-logo.png" alt="Carrot" className="carrot-logo w-12 h-12 object-contain" />
        <h2 className="carrot-title text-3xl">Consumer Dashboard</h2>
      </div>

      <div>
        <h3 className="text-2xl font-semibold mb-4 text-orange-500">Available GPUs</h3>

        {availableGPUs.length > 0 && (
          <div className="mb-4 bg-gray-50 p-4 rounded-none border-2 border-orange-500">
            <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
              <input type="text" placeholder="Search GPU..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)}
                className="bg-gray-100 border-2 border-gray-300 rounded-none px-4 py-2 text-gray-900 text-sm focus:border-orange-500 focus:outline-none" />
              <input type="number" placeholder="Min VRAM (GB)" value={minVram} onChange={e => setMinVram(e.target.value)}
                className="bg-gray-100 border-2 border-gray-300 rounded-none px-4 py-2 text-gray-900 text-sm focus:border-orange-500 focus:outline-none" />
              <input type="text" placeholder="Max Price (XLM)" value={maxPrice} onChange={e => setMaxPrice(e.target.value)}
                className="bg-gray-100 border-2 border-gray-300 rounded-none px-4 py-2 text-gray-900 text-sm focus:border-orange-500 focus:outline-none" />
              <select value={sortBy} onChange={e => setSortBy(e.target.value as any)}
                className="bg-gray-100 border-2 border-gray-300 rounded-none px-4 py-2 text-gray-900 text-sm focus:border-orange-500 focus:outline-none">
                <option value="price">Sort by Price</option>
                <option value="vram">Sort by VRAM</option>
                <option value="jobs">Sort by Experience</option>
              </select>
              <button onClick={() => { setSearchTerm(''); setMinVram(''); setMaxPrice(''); setSortBy('price'); }}
                className="bg-gray-100 hover:bg-gray-200 text-orange-500 px-4 py-2 rounded-none text-sm border-2 border-orange-500">
                Clear
              </button>
            </div>
            <div className="mt-2 text-xs text-gray-500">{filteredGPUs.length} of {availableGPUs.length} GPUs</div>
          </div>
        )}

        {loading ? <div className="text-gray-600">loading from stellar...</div> : filteredGPUs.length === 0 ?
          <div className="text-gray-600">no gpus available - register one in provider dashboard</div> : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredGPUs.map(gpu => (
              <div key={gpu.id}
                className={`bg-gray-50 p-4 rounded-none border-2 cursor-pointer transition-all ${
                  selectedGpuId === gpu.id ? 'border-orange-500 shadow-lg shadow-orange-500/30' : 'border-transparent hover:border-orange-500/50'
                }`}
                onClick={() => setSelectedGpuId(gpu.id)}>
                <div className="flex justify-between items-start mb-2">
                  <h4 className="text-lg font-semibold text-orange-500">{gpu.model}</h4>
                  <span className="px-2 py-1 rounded-none text-xs bg-orange-600 border border-orange-400">Online</span>
                </div>
                <div className="text-sm space-y-1 text-gray-700">
                  <div>VRAM: {gpu.vramGB} GB</div>
                  <div>Price: {parseFloat(gpu.pricePerHour).toFixed(2)} XLM/hour</div>
                  <div>Jobs Completed: {gpu.totalJobs}</div>
                  <div className="text-xs text-gray-500">Provider: {gpu.provider.slice(0, 8)}...</div>
                </div>
                {selectedGpuId === gpu.id && <div className="mt-2 text-sm text-orange-500 font-semibold">Selected</div>}
              </div>
            ))}
          </div>
        )}
      </div>

      {selectedGpuId !== null && (
        <div className="bg-gray-50 p-6 rounded-none border-2 border-orange-500 shadow-lg shadow-orange-500/20">
          <h3 className="text-xl font-semibold mb-4 text-orange-500">Post Job (GPU #{selectedGpuId})</h3>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Job Type</label>
              <div className="grid grid-cols-3 gap-2">
                {(['simple', 'python-script', 'docker-image'] as const).map(type => (
                  <button key={type} onClick={() => setJobType(type)}
                    className={`px-4 py-3 rounded-none border-2 transition-all ${
                      jobType === type ? 'border-orange-500 bg-orange-100 text-orange-500' : 'border-gray-300 hover:border-orange-500/50 text-gray-600'
                    }`}>
                    <div className="font-semibold capitalize">{type.replace('-', ' ')}</div>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Description</label>
              <input type="text" placeholder="what do you need computed" value={jobDescription} onChange={e => setJobDescription(e.target.value)}
                className="w-full bg-gray-100 border-2 border-gray-300 rounded-none px-4 py-2 text-gray-900 focus:border-orange-500 focus:outline-none" />
            </div>

            {jobType === 'python-script' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Python Code</label>
                <textarea value={pythonCode} onChange={e => setPythonCode(e.target.value)}
                  placeholder="import torch&#10;x = torch.randn(1000, 1000).cuda()&#10;print(f'RESULT:{x.sum().item()}')"
                  className="w-full h-64 bg-gray-100 border-2 border-gray-300 rounded-none px-4 py-2 text-gray-900 font-mono text-sm focus:border-orange-500 focus:outline-none" />
              </div>
            )}

            {jobType === 'docker-image' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Docker Image</label>
                <input type="text" value={dockerImage} onChange={e => setDockerImage(e.target.value)}
                  placeholder="username/image-name (not the url)"
                  className="w-full bg-gray-100 border-2 border-gray-300 rounded-none px-4 py-2 text-gray-900 font-mono focus:border-orange-500 focus:outline-none" />
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Compute Hours</label>
              <input type="number" placeholder="hours needed" value={computeHours} onChange={e => setComputeHours(e.target.value)}
                className="w-full bg-gray-100 border-2 border-gray-300 rounded-none px-4 py-2 text-gray-900 focus:border-orange-500 focus:outline-none" />
            </div>

            {selectedGpu && (
              <div className="p-4 bg-gray-100 rounded-none border-2 border-orange-500">
                <div className="text-sm text-gray-600 mb-2">Payment Breakdown</div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="text-xs text-gray-500">Rate</div>
                    <div className="text-gray-900 font-semibold">{parseFloat(selectedGpu.pricePerHour).toFixed(2)} XLM/hr</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Hours</div>
                    <div className="text-gray-900 font-semibold">{computeHours || '0'}h</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Provider Gets (95%)</div>
                    <div className="text-orange-500 font-semibold">{(parseFloat(calculatedPayment) * 0.95).toFixed(2)} XLM</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Platform Fee (5%)</div>
                    <div className="text-gray-600 text-sm">{(parseFloat(calculatedPayment) * 0.05).toFixed(2)} XLM</div>
                  </div>
                </div>
                <div className="mt-3 pt-3 border-t border-gray-300 flex justify-between items-center">
                  <span className="text-gray-600">Total:</span>
                  <span className="text-2xl font-bold text-orange-500">{calculatedPayment} XLM</span>
                </div>
              </div>
            )}

            <button onClick={postJob} disabled={txPending || !computeHours || calculatedPayment === '0'}
              className="w-full bg-orange-500 hover:bg-orange-600 text-gray-900 px-6 py-3 rounded-none font-medium disabled:opacity-50 text-lg border-2 border-orange-400 transition-all">
              {txPending ? 'Processing...' : `Post Job & Pay ${calculatedPayment} XLM`}
            </button>
          </div>
        </div>
      )}

      <div>
        <h3 className="text-2xl font-semibold mb-4 text-orange-500">My Jobs</h3>
        {loading ? <div className="text-gray-600">loading...</div> :
          myJobs.filter(j => j.status !== JobStatus.Completed && j.status !== JobStatus.Cancelled).length === 0 ?
          <div className="text-gray-600">no active jobs</div> : (
          <div className="space-y-4">
            {myJobs.filter(j => j.status !== JobStatus.Completed && j.status !== JobStatus.Cancelled).map(job => {
              const jobData = parseJobData(job.description);
              const providerEarnings = (parseFloat(job.paymentAmount) * 0.95).toFixed(2);
              return (
                <div key={job.jobId} className="bg-gray-50 p-6 rounded-none border-2 border-orange-500/30 hover:border-orange-500 transition-all">
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h4 className="text-xl font-semibold text-orange-500">Job #{job.jobId}</h4>
                        <span className="px-3 py-1 rounded-none text-xs bg-gray-200 text-gray-700 border border-gray-300">{jobData.type || 'simple'}</span>
                        <span className={`px-3 py-1 rounded-none text-sm font-medium border ${
                          job.status === JobStatus.Open ? 'bg-orange-600 border-orange-400' :
                          job.status === JobStatus.Claimed ? 'bg-orange-600 border-orange-400 animate-pulse' : 'bg-gray-300 border-gray-400'
                        }`}>{getJobStatusName(job.status)}</span>
                      </div>
                      <p className="text-gray-700 text-lg mb-3">{jobData.description || job.description}</p>

                      {job.status === JobStatus.Open && (
                        <div className="mb-3 p-3 bg-orange-100 rounded-none border-2 border-orange-700 text-sm">
                          waiting for provider to claim...
                        </div>
                      )}

                      <div className="grid grid-cols-2 gap-4 mt-3 p-3 bg-gray-100 rounded-none border border-orange-500/30">
                        <div>
                          <div className="text-xs text-gray-500">Your Payment</div>
                          <div className="text-lg font-semibold text-gray-900">{parseFloat(job.paymentAmount).toFixed(2)} XLM</div>
                          <div className="text-xs text-gray-500">{job.status === JobStatus.Open ? 'in escrow' : 'locked'}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-500">Provider Gets (95%)</div>
                          <div className="text-lg font-semibold text-orange-500">{providerEarnings} XLM</div>
                        </div>
                      </div>

                      {job.resultHash && (
                        <div className="mt-3 p-3 bg-orange-50 rounded-none border border-orange-500">
                          <div className="text-xs text-gray-500 mb-1">Result:</div>
                          <span className="font-mono text-xs text-orange-500 break-all">{job.resultHash}</span>
                          {(job.resultHash.startsWith('ipfs://') || job.resultHash.startsWith('Qm') || job.resultHash.startsWith('bafy')) && (
                            <div className="mt-2">
                              <a href={getIPFSGatewayUrl(job.resultHash)} target="_blank" rel="noopener noreferrer"
                                className="bg-orange-500 hover:bg-orange-600 text-gray-900 px-4 py-2 rounded-none text-sm font-semibold inline-block">
                                View Result
                              </a>
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    <div className="ml-4">
                      {job.status === JobStatus.Open && (
                        <button onClick={() => cancelJob(job.jobId)} disabled={txPending}
                          className="bg-orange-500 hover:bg-orange-600 text-gray-900 px-6 py-3 rounded-none font-medium disabled:opacity-50 border-2 border-orange-400">
                          Cancel<br/><span className="text-xs">Get refund</span>
                        </button>
                      )}
                      {job.status === JobStatus.Claimed && (
                        <div className="text-center p-3 bg-orange-100 rounded-none border-2 border-orange-500">
                          <div className="text-orange-500 font-semibold">In Progress</div>
                          <div className="text-xs text-gray-600 mt-1">provider working</div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div>
        <h3 className="text-2xl font-semibold mb-4 text-orange-500">Job History</h3>
        {myJobs.filter(j => j.status === JobStatus.Completed).length === 0 ?
          <div className="text-gray-600">no completed jobs yet</div> : (
          <div className="space-y-4">
            {myJobs.filter(j => j.status === JobStatus.Completed).map(job => {
              const jobData = parseJobData(job.description);
              return (
                <div key={job.jobId} className="bg-gray-50 p-6 rounded-none border-2 border-gray-300 opacity-90">
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h4 className="text-xl font-semibold text-gray-600">Job #{job.jobId}</h4>
                        <span className="px-3 py-1 rounded-none text-sm font-medium bg-orange-600 border border-orange-500">Completed</span>
                      </div>
                      <p className="text-gray-600">{jobData.description || job.description}</p>
                      <div className="mt-3 text-sm text-gray-500">
                        <div>Paid: {parseFloat(job.paymentAmount).toFixed(2)} XLM</div>
                        {job.resultHash && (
                          <div className="text-orange-500 mt-2">
                            Result: <span className="font-mono text-xs">{job.resultHash.slice(0, 30)}...</span>
                            {(job.resultHash.startsWith('ipfs://') || job.resultHash.startsWith('Qm') || job.resultHash.startsWith('bafy')) && (
                              <a href={getIPFSGatewayUrl(job.resultHash)} target="_blank" rel="noopener noreferrer"
                                className="ml-2 bg-orange-500 hover:bg-orange-600 text-gray-900 px-3 py-1 rounded-none text-xs font-semibold">View Result</a>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="text-center p-3 bg-orange-100 rounded-none border-2 border-orange-500">
                      <div className="text-orange-500 font-semibold">Done</div>
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
