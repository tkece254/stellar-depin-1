#![no_std]
use soroban_sdk::{contract, contractimpl, contracttype, Address, Env, String, Vec, symbol_short};

#[contracttype]
#[derive(Clone)]
pub struct GPU {
    pub provider: Address,
    pub model: String,
    pub vram_gb: u32,
    pub price_per_hour: i128,  // in stroops, baby
    pub available: bool,
    pub total_jobs: u32,
    pub registered_at: u64,
}

#[contracttype]
pub enum DataKey {
    GPU(u32),
    ProviderGPUs(Address),
    NextGPUId,
}

#[contract]
pub struct GPURegistry;

#[contractimpl]
impl GPURegistry {
    // hell yeah, register that GPU
    pub fn register_gpu(
        env: Env,
        provider: Address,
        model: String,
        vram_gb: u32,
        price_per_hour: i128,
    ) -> u32 {
        provider.require_auth();

        if vram_gb == 0 { panic!("vram cant be zero dumbass"); }
        if price_per_hour <= 0 { panic!("free gpus? nice try"); }

        let gpu_id: u32 = env.storage().instance().get(&DataKey::NextGPUId).unwrap_or(0);

        let gpu = GPU {
            provider: provider.clone(),
            model,
            vram_gb,
            price_per_hour,
            available: true,
            total_jobs: 0,
            registered_at: env.ledger().timestamp(),
        };

        env.storage().persistent().set(&DataKey::GPU(gpu_id), &gpu);

        let mut provider_gpus: Vec<u32> = env.storage().persistent()
            .get(&DataKey::ProviderGPUs(provider.clone()))
            .unwrap_or(Vec::new(&env));
        provider_gpus.push_back(gpu_id);
        env.storage().persistent().set(&DataKey::ProviderGPUs(provider), &provider_gpus);

        env.storage().instance().set(&DataKey::NextGPUId, &(gpu_id + 1));

        env.events().publish((symbol_short!("register"),), gpu_id);

        gpu_id
    }

    // flip that switch
    pub fn set_availability(env: Env, provider: Address, gpu_id: u32, available: bool) {
        provider.require_auth();

        let mut gpu: GPU = env.storage().persistent()
            .get(&DataKey::GPU(gpu_id))
            .expect("gpu doesnt exist lol");

        if gpu.provider != provider { panic!("not your gpu buddy"); }

        gpu.available = available;
        env.storage().persistent().set(&DataKey::GPU(gpu_id), &gpu);

        env.events().publish((symbol_short!("avail"),), (gpu_id, available));
    }

    // money talks
    pub fn update_price(env: Env, provider: Address, gpu_id: u32, new_price: i128) {
        provider.require_auth();

        if new_price <= 0 { panic!("price gotta be positive chief"); }

        let mut gpu: GPU = env.storage().persistent()
            .get(&DataKey::GPU(gpu_id))
            .expect("gpu where?");

        if gpu.provider != provider { panic!("hands off someone elses gpu"); }

        gpu.price_per_hour = new_price;
        env.storage().persistent().set(&DataKey::GPU(gpu_id), &gpu);
    }

    // marketplace calls this, dont touch
    pub fn increment_job_count(env: Env, gpu_id: u32) {
        let mut gpu: GPU = env.storage().persistent()
            .get(&DataKey::GPU(gpu_id))
            .expect("no gpu no party");

        gpu.total_jobs += 1;
        env.storage().persistent().set(&DataKey::GPU(gpu_id), &gpu);
    }

    // getters - the boring but necessary stuff
    pub fn get_gpu(env: Env, gpu_id: u32) -> GPU {
        env.storage().persistent()
            .get(&DataKey::GPU(gpu_id))
            .expect("gpu not found rip")
    }

    pub fn get_provider_gpus(env: Env, provider: Address) -> Vec<u32> {
        env.storage().persistent()
            .get(&DataKey::ProviderGPUs(provider))
            .unwrap_or(Vec::new(&env))
    }

    pub fn get_next_gpu_id(env: Env) -> u32 {
        env.storage().instance().get(&DataKey::NextGPUId).unwrap_or(0)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use soroban_sdk::testutils::Address as _;

    #[test]
    fn test_register_gpu() {
        let env = Env::default();
        env.mock_all_auths();
        let contract_id = env.register_contract(None, GPURegistry);
        let client = GPURegistryClient::new(&env, &contract_id);

        let provider = Address::generate(&env);
        let model = String::from_str(&env, "RTX 4090");

        let gpu_id = client.register_gpu(&provider, &model, &24, &1000000);
        assert_eq!(gpu_id, 0);

        let gpu = client.get_gpu(&0);
        assert_eq!(gpu.model, model);
        assert_eq!(gpu.vram_gb, 24);
    }
}
