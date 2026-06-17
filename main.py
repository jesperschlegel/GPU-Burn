import argparse
import copy
import signal
import threading
import time

import pynvml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

DEFAULT_MODEL_ID = "google/gemma-4-12b-it"
DEFAULT_INFERENCES_PER_ROUND = 30
DEFAULT_PREFIX_TOKENS = 4096
DEFAULT_POSTFIX_TOKENS = 64
DEFAULT_SLEEP_SECONDS = 1.0
DEFAULT_LOG_INTERVAL_SECONDS = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU burn-in via continuous LLM inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model to load."
    )
    parser.add_argument(
        "--inferences-per-round",
        type=int,
        default=DEFAULT_INFERENCES_PER_ROUND,
        help="Decode passes per prefix before rebuilding it.",
    )
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        default=DEFAULT_PREFIX_TOKENS,
        help="Length of the prefill that builds the KV cache.",
    )
    parser.add_argument(
        "--postfix-tokens",
        type=int,
        default=DEFAULT_POSTFIX_TOKENS,
        help="Tokens decoded per inference against the cached prefix.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Idle pause between rounds (0 for maximum load).",
    )
    parser.add_argument(
        "--log-interval-seconds",
        type=float,
        default=DEFAULT_LOG_INTERVAL_SECONDS,
        help="How often the logging thread samples GPU load.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds (default: run until Ctrl-C).",
    )
    return parser.parse_args()


def random_ids(n, tokenizer, device):
    return torch.randint(64, tokenizer.vocab_size, (1, n), device=device)


def get_nvml_handles():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available; a GPU is required.")
    pynvml.nvmlInit()
    return [
        pynvml.nvmlDeviceGetHandleByIndex(i)
        for i in range(pynvml.nvmlDeviceGetCount())
    ]


def log_gpu_info(handles):
    driver = pynvml.nvmlSystemGetDriverVersion()
    print(f"Driver {driver} | CUDA {torch.version.cuda} | {len(handles)} device(s)")
    for i, handle in enumerate(handles):
        name = pynvml.nvmlDeviceGetName(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        cores = pynvml.nvmlDeviceGetNumGpuCores(handle)
        tdp = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000  # mW -> W
        print(
            f"  [{i}] {name} | "
            f"{mem.total / 1024**3:.1f} GiB | "
            f"compute capability {major}.{minor} | "
            f"{cores} cores | "
            f"{tdp:.0f} W TDP"
        )


def log_gpu_load(handles):
    parts = []
    for i, handle in enumerate(handles):
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW -> W
        power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
        parts.append(
            f"[{i}] {util.gpu:3d}% util | "
            f"{mem.used / 1024**3:5.1f}/{mem.total / 1024**3:.1f} GiB | "
            f"{power:5.1f}/{power_limit:.0f} W"
        )
    print(f"{time.strftime('%H:%M:%S')}  " + "  ".join(parts), flush=True)


def logging_loop(handles, stop_event, interval):
    while not stop_event.wait(interval):
        log_gpu_load(handles)


@torch.inference_mode()
def main(args):
    handles = get_nvml_handles()
    log_gpu_info(handles)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    print(f"Loaded {args.model_id} on {model.device}. Ctrl-C to stop.")

    stop_event = threading.Event()

    def request_stop(_signum, _frame):
        print("\nStop requested; finishing current work...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    logger = threading.Thread(
        target=logging_loop,
        args=(handles, stop_event, args.log_interval_seconds),
        daemon=True,
    )
    logger.start()

    deadline = None if args.duration is None else time.monotonic() + args.duration
    try:
        while not stop_event.is_set() and (
            deadline is None or time.monotonic() < deadline
        ):
            prefix_ids = random_ids(args.prefix_tokens, tokenizer, model.device)
            prefix_cache = DynamicCache()
            model(prefix_ids, past_key_values=prefix_cache, use_cache=True)

            for _ in range(args.inferences_per_round):
                if stop_event.is_set():
                    break
                cache = copy.deepcopy(prefix_cache)
                postfix_ids = random_ids(args.postfix_tokens, tokenizer, model.device)
                attention_mask = torch.ones(
                    (1, args.prefix_tokens + args.postfix_tokens), device=model.device
                )
                model(
                    postfix_ids,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                )
            time.sleep(args.sleep_seconds)
    finally:
        stop_event.set()
        logger.join()


if __name__ == "__main__":
    main(parse_args())
