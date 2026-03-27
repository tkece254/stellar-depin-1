export interface GPU {
  provider: string;
  model: string;
  vramGB: number;
  pricePerHour: string;
  available: boolean;
  totalJobs: number;
  registeredAt: number;
}

export const JobStatus = {
  Open: 0,
  Claimed: 1,
  Completed: 2,
  Cancelled: 3,
} as const;

export type JobStatusType = typeof JobStatus[keyof typeof JobStatus];

export function getJobStatusName(status: number): string {
  const names: Record<number, string> = {
    0: 'Open',
    1: 'Claimed',
    2: 'Completed',
    3: 'Cancelled',
  };
  return names[status] || 'wtf';
}

export interface Job {
  jobId: number;
  consumer: string;
  gpuId: number;
  description: string;
  computeHours: number;
  paymentAmount: string;
  provider: string;
  status: number;
  createdAt: number;
  claimedAt: number;
  completedAt: number;
  resultHash: string;
}
