# Carrot - GPU Rental on Stellar

decentralized gpu rental. pay in xlm. no middlemen. simple.

## what is this

rent gpus or make money from your idle hardware. built on stellar (soroban smart contracts).

## structure

```
carrot/
├── src/                # react frontend
│   ├── config/         # contract addresses & helpers
│   ├── hooks/          # wallet hook
│   ├── pages/          # provider & consumer dashboards
│   ├── types/          # typescript types
│   └── utils/          # ipfs helpers
├── contracts/          # soroban smart contracts (rust)
│   ├── gpu_registry/   # register and manage gpus
│   └── job_marketplace/ # post jobs, escrow, payments
├── worker/             # gpu execution server
├── scripts/            # deploy scripts
└── examples/           # docker & script examples
```

## quick start

```bash
npm install
npm run dev
```

## all commands

| Command | What it does |
|---------|-------------|
| `npm run dev` | Start frontend dev server |
| `npm run build` | Build for production |
| `npm run worker` | Start GPU worker server |
| `npm run contracts:build` | Build Soroban contracts |
| `npm run contracts:test` | Run contract tests |
| `npm run contracts:deploy` | Build + deploy + initialize contracts |

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

create a `.env` file at the root:
```
VITE_PINATA_JWT=your_jwt_here
PINATA_JWT=your_jwt_here
```

## deployed contracts (testnet)

- GPU Registry: `CAHUSLBSYQETDXGUEDPMA6QNLRJBX2JW7BYL2LXSWYBLZN543FKM5LAX`
- Job Marketplace: `CAWNKF4EFHJYJPHZSIXYPG5D45HFGLBNFN6NGO3676YTRLNNXGKJ7PMW`

## colors

- primary: orange-500 (#f97316)
- bg: white
- accent: orange-400

---

built different
