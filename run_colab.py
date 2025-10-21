import argparse
import os
from rag_pipeline import RAGResources, env
from dataset_loader import ingest_all
import subprocess
import sys


def ensure_dirs():
    os.makedirs(env("CHROMA_DIR", "./chroma"), exist_ok=True)
    os.makedirs(env("WORKSPACE_DIR", "./workspace"), exist_ok=True)

def ingest_cli(args):
    ensure_dirs()
    resources = RAGResources()
    result = ingest_all(
        resources,
        swe_max=int(env("SWE_MAX_DOCS", str(args.swe_max))),
        stack_max=int(env("STACK_MAX_DOCS", str(args.stack_max))),
        rstar_max=int(env("RSTAR_MAX_DOCS", str(args.rstar_max))),
        stack_langs=[l.strip() for l in env("THE_STACK_LANGS", args.stack_langs).split(",") if l.strip()],
    )
    print(result)

def serve_cli(args):
    ensure_dirs()
    envs = os.environ.copy()
    cmd = [sys.executable, "-m", "uvicorn", "api_server:app", "--host", env("API_HOST", "0.0.0.0"), "--port", env("API_PORT", "8000")]
    if args.reload:
        cmd.append("--reload")
    subprocess.run(cmd, env=envs, check=True)

def all_cli(args):
    ingest_cli(args)
    serve_cli(args)

def main():
    parser = argparse.ArgumentParser(description="WB AI Core runner")
    parser.add_argument("--swe-max", type=int, default=500)
    parser.add_argument("--stack-max", type=int, default=200)
    parser.add_argument("--rstar-max", type=int, default=200)
    parser.add_argument("--stack-langs", type=str, default="python,javascript")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("mode", choices=["ingest", "serve", "all"], help="Run mode")
    args = parser.parse_args()

    if args.mode == "ingest":
        ingest_cli(args)
    elif args.mode == "serve":
        serve_cli(args)
    else:
        all_cli(args)

if __name__ == "__main__":
    main()
