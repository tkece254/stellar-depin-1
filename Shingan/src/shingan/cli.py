from __future__ import annotations

import argparse
import json
import secrets
import time
from dataclasses import asdict
from pathlib import Path

import torch

from shingan.attacks import attacks_from_json
from shingan.bench import load_manifest, run_campaign, run_corpus, summarize_campaign, summarize_corpus
from shingan.block import BLOCK_OPS, BlockConfig, Ops, TransformerBlock, evaluate
from shingan.breaker import break_kernel
from shingan.builder import build
from shingan.contracts import Verdict
from shingan.isolation import attack_isolated, verify_isolated
from shingan.patch import KernelTable
from shingan.pipeline import apply_attacks, attack, standard, verify
from shingan.problems import PROBLEMS, dtype_name, get_problem, parse_dtype
from shingan.replay import replay_gate
from shingan.tokenfactory import ROLE_KEYWORDS, TokenFactory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shingan", description="Verified speedups for AI-written GPU kernels")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("problems", help="list built-in problems")
    commands.add_parser("models", help="list Token Factory models and the Nemotron model chosen per role")

    verify_cmd = commands.add_parser("verify", help="verify a kernel in an isolated worker")
    _add_kernel_args(verify_cmd)
    verify_cmd.add_argument("--out", type=Path, default=Path("verdict.json"))
    attack_source = verify_cmd.add_mutually_exclusive_group()
    attack_source.add_argument("--gates-only", action="store_true", help="skip the Breaker")
    attack_source.add_argument("--attacks", type=Path, help="replay saved attacks instead of asking the Breaker")

    worker_cmd = commands.add_parser("worker", help="verify a kernel in this process")
    _add_kernel_args(worker_cmd)
    worker_cmd.add_argument("--out", type=Path, required=True)
    worker_cmd.add_argument("--mode", choices=["verify", "standard", "attack"], default="verify")
    worker_cmd.add_argument("--attacks", type=Path, help="attacks to run in attack mode")

    build_cmd = commands.add_parser("build", help="have Nemotron write a kernel, verify it and repair it")
    build_cmd.add_argument("--problem", required=True, choices=sorted(PROBLEMS))
    build_cmd.add_argument("--dtype", default="float16")
    build_cmd.add_argument("--attempts", type=int, default=6)
    build_cmd.add_argument("--runs", type=Path, default=Path("runs"))

    bench_cmd = commands.add_parser("bench", help="measure Shingan on a kernel corpus or a Builder campaign")
    suite = bench_cmd.add_mutually_exclusive_group(required=True)
    suite.add_argument("--manifest", type=Path, help="kernel corpus manifest (Suites A and A')")
    suite.add_argument("--problems", help="comma-separated problems for a Builder campaign (Suite B)")
    bench_cmd.add_argument("--breaker", choices=["live", "replay", "off"], default="live")
    bench_cmd.add_argument("--replay", type=Path, help="earlier bench output whose attacks to replay")
    bench_cmd.add_argument("--dtype", default="float16")
    bench_cmd.add_argument("--attempts", type=int, default=6)
    bench_cmd.add_argument("--out", type=Path, default=Path("runs") / f"bench-{time.strftime('%Y%m%d-%H%M%S')}")

    replay_cmd = commands.add_parser("replay", help="re-run one failing case from a verdict and show the worst elements")
    replay_cmd.add_argument("kernel", type=Path)
    replay_cmd.add_argument("--verdict", type=Path, required=True)
    replay_cmd.add_argument("--gate", required=True, help="gate name from the verdict, such as G2 or B:long-rows")
    replay_cmd.add_argument("--attacks", type=Path, help="attacks file, defaults to attacks.json beside the verdict")

    block_cmd = commands.add_parser("block", help="patch verified kernels into a transformer block and measure it (Suite C)")
    block_cmd.add_argument("--runs", type=Path, nargs="*", default=[], help="build run directories to take GENUINE kernels from")
    block_cmd.add_argument("--dtype", default="float16")
    block_cmd.add_argument("--seq", type=int, default=BlockConfig.seq)
    block_cmd.add_argument("--hidden", type=int, default=BlockConfig.hidden)
    block_cmd.add_argument("--heads", type=int, default=BlockConfig.heads)
    block_cmd.add_argument("--ffn", type=int, default=BlockConfig.ffn)
    block_cmd.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    match args.command:
        case "problems":
            return _problems()
        case "models":
            return _models()
        case "verify":
            return _verify(args)
        case "worker":
            return _worker(args)
        case "build":
            return _build(args)
        case "bench":
            return _bench(args)
        case "block":
            return _block(args)
        case "replay":
            return _replay(args)
    return 2


def _add_kernel_args(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("kernel", type=Path)
    cmd.add_argument("--problem", required=True, choices=sorted(PROBLEMS))
    cmd.add_argument("--dtype", default="float16")
    cmd.add_argument("--seed", type=int)


def _problems() -> int:
    for problem in PROBLEMS.values():
        args = ", ".join(f"{a.name}[{','.join(a.dims)}]" for a in problem.args)
        print(f"{problem.name:<10} {args}")
    return 0


def _models() -> int:
    client = TokenFactory.from_env()
    for model in client.models():
        print(model)
    print()
    for role in ROLE_KEYWORDS:
        print(f"{role:<8} -> {client.model_for(role)}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    problem = get_problem(args.problem)
    dtype = dtype_name(parse_dtype(args.dtype))
    verdict = verify_isolated(problem.name, args.kernel, dtype, args.out, seed=args.seed)
    if args.attacks:
        results_path = args.out.with_name(f"{args.out.stem}-attack-results.json")
        verdict = apply_attacks(verdict, attack_isolated(problem.name, args.kernel, dtype, args.attacks, results_path))
    elif not args.gates_only:
        verdict, _ = break_kernel(TokenFactory.from_env(), problem, args.kernel, dtype, verdict, args.out.parent)
    args.out.write_text(json.dumps(verdict.to_dict(), indent=2))
    print(render(verdict))
    return 0


def _worker(args: argparse.Namespace) -> int:
    problem = get_problem(args.problem)
    dtype = parse_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed if args.seed is not None else secrets.randbits(31)
    match args.mode:
        case "verify":
            output: object = verify(problem, args.kernel, dtype, device, seed).to_dict()
        case "standard":
            output = asdict(standard(problem, args.kernel, dtype, device, seed))
        case "attack":
            if args.attacks is None:
                raise SystemExit("--mode attack needs --attacks")
            attacks = attacks_from_json(args.attacks.read_text())
            output = [asdict(r) for r in attack(problem, args.kernel, dtype, device, seed, attacks)]
    args.out.write_text(json.dumps(output, indent=2))
    return 0


def _build(args: argparse.Namespace) -> int:
    dtype = dtype_name(parse_dtype(args.dtype))
    run_dir = args.runs / f"{args.problem}-{dtype}-{time.strftime('%Y%m%d-%H%M%S')}"
    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    run = build(get_problem(args.problem), dtype, TokenFactory.from_env(), run_dir, args.attempts, device_name)
    for attempt in run.attempts:
        status = render_line(attempt.verdict) if attempt.verdict else "no code in reply"
        print(f"attempt {attempt.index}: {status}")
    for model, totals in run.usage.items():
        print(f"{model}: {totals.prompt} prompt + {totals.completion} completion tokens")
    print(f"run saved to {run_dir}")
    return 0


def _bench(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        client = TokenFactory.from_env() if args.breaker == "live" else None
        results, usage = run_corpus(load_manifest(args.manifest), args.out, args.breaker, client, args.replay)
        summary = asdict(summarize_corpus(results))
        report = {"summary": summary, "entries": [r.to_dict() for r in results]}
        for r in results:
            marks = "  ".join(f"{c}:{'flag' if r.flagged(c) else 'pass'}" for c in ("B0", "B1", "B2"))
            label = f"cheat {r.entry.family or ''}" if r.entry.cheat else "control"
            print(f"{r.entry.id:<40} {label:<12} {marks}")
    else:
        dtype = dtype_name(parse_dtype(args.dtype))
        problems = [p.strip() for p in args.problems.split(",") if p.strip()]
        for name in problems:
            get_problem(name)
        device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else None
        outcomes, usage = run_campaign(problems, dtype, args.attempts, TokenFactory.from_env(), args.out, device_name)
        summary = asdict(summarize_campaign(outcomes))
        report = {"summary": summary, "problems": [o.to_dict() for o in outcomes]}
        for o in outcomes:
            print(f"{o.problem:<12} {render_line(o.final) if o.final else 'no verdict'}")
    report["usage"] = {model: asdict(totals) for model, totals in usage.items()}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"results saved to {args.out}")
    return 0


def _block(args: argparse.Namespace) -> int:
    dtype = parse_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = KernelTable.from_runs(args.runs)
    for (problem, kernel_dtype), verified in table.kernels.items():
        print(f"patched  {problem} [{dtype_name(kernel_dtype)}] from {verified.path}")
    config = BlockConfig(args.seq, args.hidden, args.heads, args.ffn)
    patched_ops = Ops(**{name: table.op(name) for name in BLOCK_OPS})
    report = evaluate(TransformerBlock.random(config, args.seed), patched_ops, dtype, device, args.seed)
    print("\nop time ranking (original block):")
    for name, ms in report.profile:
        print(f"  {name:<8} {ms:.3f} ms")
    print(f"\nerror vs fp64 (tolerance units): original {report.original_error:.3g}, patched {report.patched_error:.3g}")
    print(f"correct: {report.correct}")
    if report.speedup is not None:
        print(f"latency: original {report.original.median_ms:.3f} ms, patched {report.patched.median_ms:.3f} ms, {report.speedup:.2f}x")
    return 0 if report.correct else 1


def _replay(args: argparse.Namespace) -> int:
    verdict = Verdict.from_dict(json.loads(args.verdict.read_text()))
    attacks = args.attacks or args.verdict.with_name("attacks.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = replay_gate(args.kernel, verdict, args.gate, device, attacks if attacks.is_file() else None)
    print(f"{result.gate.gate} recorded: {result.gate.detail}")
    print(f"replayed: {result.mismatch or 'within tolerance'}")
    if result.error is not None:
        print(f"max error: {result.error:.3g} tolerance units")
    for d in result.worst:
        print(f"  at {list(d.index)}: expected {d.expected:.6g}, got {d.got:.6g} ({d.tolerance_units:.3g} units)")
    return 0 if result.mismatch is None else 1


def render_line(verdict: Verdict) -> str:
    parts = [verdict.outcome.value]
    if verdict.family:
        parts.append(verdict.family)
    if verdict.speedup_vs_eager is not None:
        parts.append(f"{verdict.speedup_vs_eager:.2f}x eager")
    if verdict.speedup_vs_compiled is not None:
        parts.append(f"{verdict.speedup_vs_compiled:.2f}x torch.compile")
    return "  ".join(parts)


def render(verdict: Verdict) -> str:
    lines = [f"{verdict.problem} [{verdict.dtype}]  {render_line(verdict)}"]
    if verdict.error:
        lines.append(f"  error: {verdict.error}")
    for gate in verdict.gates:
        detail = f"  {gate.detail}" if gate.detail else ""
        lines.append(f"  {gate.gate:<4} {gate.status.value:<5}{detail}")
        if gate.replay:
            lines.append(f"       replay: {json.dumps(asdict(gate.replay))}")
    for label, timing in (("kernel", verdict.candidate), ("eager", verdict.eager), ("compile", verdict.compiled)):
        if timing:
            lines.append(f"  {label:<8} {timing.median_ms:.4f} ms  (IQR {timing.q1_ms:.4f}-{timing.q3_ms:.4f})")
    return "\n".join(lines)
