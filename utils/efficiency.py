import gc
import time
from copy import deepcopy
import numpy as np
import torch
from thop import profile, clever_format


def _release_cuda(device='cuda'):
    """
    Force-free CUDA memory held by unreachable objects.

    thop.profile() attaches forward hooks (closures) directly onto every
    submodule to accumulate total_ops/total_params. Those hooks create a
    reference cycle between each module and its own hook closure, so plain
    `del` / refcounting alone does NOT guarantee the profiled copy (and any
    activation tensors captured by the hooks) are freed immediately —
    Python's cyclic GC has to run first. Without this, ~GB-scale CUDA
    memory can stay "allocated" (not just cached) across many repeated
    benchmark forward passes, silently starving the training loop that
    runs right after this report.
    """
    gc.collect()
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def measure_latency(model, input_size=(1, 3, 352, 352), device='cuda',
                    warmup=50, iterations=200):
    model.eval()
    dummy = torch.randn(input_size).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    del dummy
    _release_cuda(device)

    times = np.array(times)
    return {
        'mean_ms':  float(np.mean(times)),
        'std_ms':   float(np.std(times)),
        'p50_ms':   float(np.percentile(times, 50)),
        'p90_ms':   float(np.percentile(times, 90)),
        'p99_ms':   float(np.percentile(times, 99)),
        'min_ms':   float(np.min(times)),
        'max_ms':   float(np.max(times)),
        'fps':      float(1000.0 / np.mean(times)),
    }


def measure_throughput(model, input_size=(1, 3, 352, 352), device='cuda',
                       batch_sizes=(1, 4, 8, 16), warmup=10, iterations=50):
    model.eval()
    results = {}
    for bs in batch_sizes:
        shape = (bs,) + input_size[1:]
        dummy = torch.randn(shape).to(device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)
                if device == 'cuda':
                    torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                if device == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy)
                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        avg = np.mean(times)
        results[bs] = {
            'latency_ms': float(avg * 1000),
            'fps':        float(bs / avg),
        }
        del dummy
        _release_cuda(device)
    return results


def measure_peak_memory(model, input_size=(1, 3, 352, 352), device='cuda'):
    if device != 'cuda' or not torch.cuda.is_available():
        return {'peak_mb': 0.0, 'allocated_mb': 0.0}

    model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    dummy = torch.randn(input_size).to(device)

    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    allocated_mb = torch.cuda.memory_allocated() / 1024**2

    del dummy
    _release_cuda(device)

    return {
        'peak_mb':      peak_mb,
        'allocated_mb': allocated_mb,
    }


def measure_flops_params(model, input_size=(1, 3, 352, 352)):
    device = next(model.parameters()).device
    profile_model = deepcopy(model).to(device)
    dummy = torch.randn(input_size).to(device)
    profile_model.eval()
    with torch.no_grad():
        flops, params = profile(profile_model, inputs=(dummy,), verbose=False)
    del profile_model, dummy
    # thop's hooks create a module<->closure reference cycle — force the
    # cyclic collector to run now instead of leaving it to chance.
    _release_cuda(device)
    flops_str, params_str = clever_format([flops, params], '%.3f')
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024**2
    return {
        'gflops':           float(flops / 1e9),
        'flops_str':        flops_str,
        'total_params':     total_params,
        'trainable_params': trainable_params,
        'params_str':       params_str,
        'model_size_mb':    float(model_size_mb),
    }


def measure_energy(model, input_size=(1, 3, 352, 352), device='cuda',
                   warmup=10, iterations=100):
    model.eval()
    dummy = torch.randn(input_size).to(device)

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)
                if device == 'cuda':
                    torch.cuda.synchronize()

        power_readings = []
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                if device == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_readings.append(power_mw / 1000.0)
                times.append(elapsed)

        avg_power_w = float(np.mean(power_readings))
        avg_time_s = float(np.mean(times))
        energy_mj = avg_power_w * avg_time_s * 1000.0

        del dummy
        _release_cuda(device)

        return {
            'avg_power_w':  avg_power_w,
            'energy_mj':    energy_mj,
            'method':       'pynvml',
        }

    except Exception:
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)
                if device == 'cuda':
                    torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(iterations):
                if device == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy)
                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        avg_time_s = float(np.mean(times))

        del dummy
        _release_cuda(device)

        return {
            'avg_power_w':  None,
            'energy_mj':    None,
            'avg_time_s':   avg_time_s,
            'method':       'time_only',
        }


def full_efficiency_report(model, input_size=(1, 3, 352, 352), device='cuda'):
    device_str = device if isinstance(device, str) else ('cuda' if device.type == 'cuda' else 'cpu')

    flops = measure_flops_params(model, input_size)
    latency = measure_latency(model, input_size, device_str)
    memory = measure_peak_memory(model, input_size, device_str)
    energy = measure_energy(model, input_size, device_str)

    print(f'\n{"="*60}')
    print('EFFICIENCY REPORT')
    print(f'{"="*60}')
    print(f'GFLOPs:          {flops["gflops"]:.4f}')
    print(f'Params:          {flops["params_str"]}  ({flops["total_params"]:,})')
    print(f'Model Size:      {flops["model_size_mb"]:.2f} MB')
    print(f'Peak GPU Mem:    {memory["peak_mb"]:.2f} MB')
    print(f'Latency mean:    {latency["mean_ms"]:.3f} ± {latency["std_ms"]:.3f} ms')
    print(f'Latency P50:     {latency["p50_ms"]:.3f} ms')
    print(f'Latency P90:     {latency["p90_ms"]:.3f} ms')
    print(f'Latency P99:     {latency["p99_ms"]:.3f} ms')
    print(f'FPS:             {latency["fps"]:.2f}')
    if energy['energy_mj'] is not None:
        print(f'Avg Power:       {energy["avg_power_w"]:.2f} W')
        print(f'Energy/inf:      {energy["energy_mj"]:.4f} mJ')
    print(f'{"="*60}\n')

    # Final sweep: after ~350+ benchmark forward passes + one deepcopy/thop
    # profiling pass, make sure nothing lingers before the caller hands
    # `model` straight into the training loop.
    _release_cuda(device_str)
    if device_str == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    return {**flops, **latency, **memory, **energy}
