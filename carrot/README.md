# Carrot - GPU Rental on Stellar

decentralized gpu rental. pay in xlm. no middlemen. simple.

## what is this

rent gpus or make money from your idle hardware. built on stellar (soroban smart contracts).

## structure

```
carrot/
├── contracts/          # soroban smart contracts (rust)
│   ├── gpu_registry/   # register and manage gpus
│   └── job_marketplace/ # post jobs, escrow, payments
├── frontend/           # react app
└── provider-worker/    # gpu execution server
```

## quick start

### frontend
```bash
cd frontend
npm install
npm run dev
```

### provider worker
```bash
cd provider-worker
npm install
npm start
```

### contracts
```bash
cd contracts
cargo build --release --target wasm32-unknown-unknown
```

## how it works

1. **providers** register gpus with price per hour
2. **consumers** browse gpus and post jobs with xlm payment
3. payment locked in contract (escrow)
4. provider claims job, runs compute
5. provider submits result, gets 95% of payment
6. platform keeps 5%

## tech stack

- **chain**: stellar (soroban)
- **contracts**: rust
- **frontend**: react + typescript + tailwind
- **wallet**: freighter
- **storage**: ipfs (pinata)
- **gpu isolation**: docker

## env vars

frontend `.env`:
```
VITE_PINATA_JWT=your_jwt_here
```

provider-worker `.env`:
```
PINATA_JWT=your_jwt_here
```

## colors

- primary: orange-500 (#f97316)
- bg: zinc-950
- accent: orange-400

## deploy contracts

stellar cli needed. check stellar docs for soroban deployment.

---

built different
