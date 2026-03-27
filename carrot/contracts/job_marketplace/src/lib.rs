#![no_std]
use soroban_sdk::{contract, contractimpl, contracttype, token, Address, Env, String, Vec, symbol_short};

#[contracttype]
#[derive(Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum JobStatus {
    Open = 0,
    Claimed = 1,
    Completed = 2,
    Cancelled = 3,
}

#[contracttype]
#[derive(Clone)]
pub struct Job {
    pub job_id: u32,
    pub consumer: Address,
    pub gpu_id: u32,
    pub description: String,
    pub compute_hours: u32,
    pub payment_amount: i128,  // XLM in stroops
    pub provider: Address,
    pub status: JobStatus,
    pub created_at: u64,
    pub claimed_at: u64,
    pub completed_at: u64,
    pub result_hash: String,
}

#[contracttype]
pub enum DataKey {
    Job(u32),
    ConsumerJobs(Address),
    ProviderJobs(Address),
    NextJobId,
    GPURegistry,
    TokenAddress,
    PlatformFees,
}

const PLATFORM_FEE_PERCENT: i128 = 5;

#[contract]
pub struct JobMarketplace;

#[contractimpl]
impl JobMarketplace {
    // init this bad boy
    pub fn initialize(env: Env, gpu_registry: Address, token_address: Address) {
        if env.storage().instance().has(&DataKey::GPURegistry) {
            panic!("already initialized genius");
        }
        env.storage().instance().set(&DataKey::GPURegistry, &gpu_registry);
        env.storage().instance().set(&DataKey::TokenAddress, &token_address);
        env.storage().instance().set(&DataKey::PlatformFees, &0i128);
    }

    // lets fucking go - post a job and lock that payment
    pub fn post_job(
        env: Env,
        consumer: Address,
        gpu_id: u32,
        description: String,
        compute_hours: u32,
        payment_amount: i128,
    ) -> u32 {
        consumer.require_auth();

        if compute_hours == 0 { panic!("0 hours? what are you doing"); }
        if payment_amount <= 0 { panic!("gotta pay something bro"); }

        // yoink the payment into escrow
        let token_address: Address = env.storage().instance().get(&DataKey::TokenAddress).unwrap();
        let token_client = token::Client::new(&env, &token_address);
        token_client.transfer(&consumer, &env.current_contract_address(), &payment_amount);

        let job_id: u32 = env.storage().instance().get(&DataKey::NextJobId).unwrap_or(0);

        let job = Job {
            job_id,
            consumer: consumer.clone(),
            gpu_id,
            description,
            compute_hours,
            payment_amount,
            provider: consumer.clone(), // placeholder until claimed
            status: JobStatus::Open,
            created_at: env.ledger().timestamp(),
            claimed_at: 0,
            completed_at: 0,
            result_hash: String::from_str(&env, ""),
        };

        env.storage().persistent().set(&DataKey::Job(job_id), &job);

        let mut consumer_jobs: Vec<u32> = env.storage().persistent()
            .get(&DataKey::ConsumerJobs(consumer.clone()))
            .unwrap_or(Vec::new(&env));
        consumer_jobs.push_back(job_id);
        env.storage().persistent().set(&DataKey::ConsumerJobs(consumer), &consumer_jobs);

        env.storage().instance().set(&DataKey::NextJobId, &(job_id + 1));

        env.events().publish((symbol_short!("posted"),), job_id);

        job_id
    }

    // provider claims the bag
    pub fn claim_job(env: Env, provider: Address, job_id: u32) {
        provider.require_auth();

        let mut job: Job = env.storage().persistent()
            .get(&DataKey::Job(job_id))
            .expect("job doesnt exist");

        if job.status != JobStatus::Open { panic!("job aint open"); }

        job.provider = provider.clone();
        job.status = JobStatus::Claimed;
        job.claimed_at = env.ledger().timestamp();

        env.storage().persistent().set(&DataKey::Job(job_id), &job);

        let mut provider_jobs: Vec<u32> = env.storage().persistent()
            .get(&DataKey::ProviderJobs(provider.clone()))
            .unwrap_or(Vec::new(&env));
        provider_jobs.push_back(job_id);
        env.storage().persistent().set(&DataKey::ProviderJobs(provider), &provider_jobs);

        env.events().publish((symbol_short!("claimed"),), job_id);
    }

    // show me the money
    pub fn complete_job(env: Env, provider: Address, job_id: u32, result_hash: String) {
        provider.require_auth();

        let mut job: Job = env.storage().persistent()
            .get(&DataKey::Job(job_id))
            .expect("wtf wheres the job");

        if job.status != JobStatus::Claimed { panic!("cant complete unclaimed job"); }
        if job.provider != provider { panic!("you didnt claim this job homie"); }

        job.status = JobStatus::Completed;
        job.completed_at = env.ledger().timestamp();
        job.result_hash = result_hash;

        // split that cash - 95% to provider, 5% platform
        let platform_fee = (job.payment_amount * PLATFORM_FEE_PERCENT) / 100;
        let provider_payment = job.payment_amount - platform_fee;

        let token_address: Address = env.storage().instance().get(&DataKey::TokenAddress).unwrap();
        let token_client = token::Client::new(&env, &token_address);

        // pay the provider
        token_client.transfer(&env.current_contract_address(), &provider, &provider_payment);

        // stack them platform fees
        let mut fees: i128 = env.storage().instance().get(&DataKey::PlatformFees).unwrap_or(0);
        fees += platform_fee;
        env.storage().instance().set(&DataKey::PlatformFees, &fees);

        env.storage().persistent().set(&DataKey::Job(job_id), &job);

        env.events().publish((symbol_short!("done"),), (job_id, provider_payment));
    }

    // consumer wants their money back
    pub fn cancel_job(env: Env, consumer: Address, job_id: u32) {
        consumer.require_auth();

        let mut job: Job = env.storage().persistent()
            .get(&DataKey::Job(job_id))
            .expect("no job no cry");

        if job.consumer != consumer { panic!("not your job to cancel"); }

        if job.status == JobStatus::Claimed {
            let time_elapsed = env.ledger().timestamp() - job.claimed_at;
            if time_elapsed < 86400 { panic!("chill, wait 24h before canceling claimed job"); }
        } else if job.status != JobStatus::Open {
            panic!("too late to cancel this one");
        }

        job.status = JobStatus::Cancelled;

        // refund the homie
        let token_address: Address = env.storage().instance().get(&DataKey::TokenAddress).unwrap();
        let token_client = token::Client::new(&env, &token_address);
        token_client.transfer(&env.current_contract_address(), &consumer, &job.payment_amount);

        env.storage().persistent().set(&DataKey::Job(job_id), &job);

        env.events().publish((symbol_short!("cancel"),), job_id);
    }

    // boring getters
    pub fn get_job(env: Env, job_id: u32) -> Job {
        env.storage().persistent()
            .get(&DataKey::Job(job_id))
            .expect("job not found sadge")
    }

    pub fn get_consumer_jobs(env: Env, consumer: Address) -> Vec<u32> {
        env.storage().persistent()
            .get(&DataKey::ConsumerJobs(consumer))
            .unwrap_or(Vec::new(&env))
    }

    pub fn get_provider_jobs(env: Env, provider: Address) -> Vec<u32> {
        env.storage().persistent()
            .get(&DataKey::ProviderJobs(provider))
            .unwrap_or(Vec::new(&env))
    }

    pub fn get_next_job_id(env: Env) -> u32 {
        env.storage().instance().get(&DataKey::NextJobId).unwrap_or(0)
    }

    pub fn get_platform_fees(env: Env) -> i128 {
        env.storage().instance().get(&DataKey::PlatformFees).unwrap_or(0)
    }
}
