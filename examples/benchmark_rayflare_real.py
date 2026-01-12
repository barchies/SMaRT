#!/usr/bin/env python3
"""
Real RayFlare benchmark - measure actual computation time.

Author: D. Barchiesi / Claude Code
Date: January 2026
"""

import numpy as np
import time
import sys
from pathlib import Path
import shutil
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Clean previous results
rf_results = Path.home() / 'RayFlare_results'
if rf_results.exists():
    shutil.rmtree(rf_results)

print("="*70)
print("REAL RAYFLARE BENCHMARK")
print("="*70)

# Import RayFlare
from rayflare.textures import regular_pyramids
from rayflare.structure import Interface, BulkLayer, Structure
from rayflare.matrix_formalism import process_structure, calculate_RAT
from rayflare.options import default_options
from solcore import material, si
from solcore.structure import Layer

# Materials
print("\nSetting up materials...")
Si = material('Si')()
SiN = material('Si3N4')()
Air = material('Air')()

# Structure matching Vogt 2022
print("Setting up structure: Air / SiNx(75nm) / Si(180µm) textured / Al")

# Pyramid texture
pyramid = regular_pyramids(elevation_angle=54.74, upright=True)

# Test configurations
test_configs = [
    {'n_wl': 10, 'n_rays': 500, 'name': '10 λ, 500 rays'},
    {'n_wl': 20, 'n_rays': 500, 'name': '20 λ, 500 rays'},
    {'n_wl': 50, 'n_rays': 500, 'name': '50 λ, 500 rays'},
]

results = []

for cfg in test_configs:
    print(f"\n{'='*60}")
    print(f"TEST: {cfg['name']}")
    print(f"{'='*60}")

    # Clean results
    if rf_results.exists():
        shutil.rmtree(rf_results)

    # Wavelengths
    wl = np.linspace(400, 1100, cfg['n_wl']) * 1e-9

    # Options
    opts = default_options()
    opts.wavelengths = wl
    opts.project_name = f"bench_{cfg['n_wl']}"
    opts.n_rays = cfg['n_rays']
    opts.nx = 5
    opts.ny = 5

    # Structure
    front = Interface('RT_TMM', texture=pyramid,
                      layers=[Layer(si('75nm'), SiN)], name='front')
    bulk = BulkLayer(si('180um'), Si, name='Si')
    back = Interface('TMM', layers=[], name='back')
    struct = Structure([front, bulk, back], incidence=Air, transmission=Si)

    # Benchmark
    print(f"  Running RayFlare ({cfg['n_wl']} wavelengths, {cfg['n_rays']} rays)...")

    t0 = time.perf_counter()
    process_structure(struct, opts)
    t_process = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = calculate_RAT(struct, opts)
    t_calc = time.perf_counter() - t1

    t_total = t_process + t_calc

    print(f"  Process structure: {t_process:.2f} s")
    print(f"  Calculate RAT: {t_calc:.2f} s")
    print(f"  Total: {t_total:.2f} s")
    print(f"  Per wavelength: {t_total/cfg['n_wl']:.2f} s")

    results.append({
        'config': cfg['name'],
        'n_wl': cfg['n_wl'],
        'n_rays': cfg['n_rays'],
        't_total': t_total,
        't_per_wl': t_total / cfg['n_wl']
    })

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\n{'Config':<25} {'Total (s)':<12} {'Per λ (s)':<12}")
print("-"*50)
for r in results:
    print(f"{r['config']:<25} {r['t_total']:<12.1f} {r['t_per_wl']:<12.2f}")

# Average time per wavelength
avg_t_per_wl = np.mean([r['t_per_wl'] for r in results])
print(f"\n  Average time per wavelength: {avg_t_per_wl:.2f} s")

# Compare with SMaRT
print("\n" + "="*70)
print("COMPARISON WITH SMaRT")
print("="*70)

# Load SMaRT benchmark
from config import load_config
from src.hybrid_smatrix_coupled import create_coupled_model

config = load_config('vogt_2022_iam')
model = create_coupled_model(config)

# Warmup
wl_test = np.linspace(400, 1100, 50)
_ = model.compute_absorption(wl_test[:5], theta_ext_deg=0)

# Benchmark SMaRT
print("\n  Benchmarking SMaRT (50 λ)...")
times_smart = []
for i in range(5):
    t0 = time.perf_counter()
    _ = model.compute_absorption(wl_test, theta_ext_deg=0)
    times_smart.append(time.perf_counter() - t0)

t_smart = np.mean(times_smart)
t_smart_per_wl = t_smart / len(wl_test)

print(f"  SMaRT: {t_smart*1000:.1f} ms for {len(wl_test)} λ")
print(f"  SMaRT per λ: {t_smart_per_wl*1000:.3f} ms")

# Speedup
speedup = avg_t_per_wl / t_smart_per_wl
print(f"\n  RayFlare per λ: {avg_t_per_wl:.2f} s = {avg_t_per_wl*1000:.0f} ms")
print(f"  SMaRT per λ: {t_smart_per_wl*1000:.3f} ms")
print(f"\n  >>> MEASURED SPEEDUP: {speedup:.0f}x <<<")

# Calibration scenario
n_evals = 1000
n_wl_full = 901
t_smart_calib = n_evals * n_wl_full * t_smart_per_wl
t_rf_calib = n_evals * n_wl_full * avg_t_per_wl

print(f"\n  CALIBRATION SCENARIO ({n_evals} evals × {n_wl_full} λ):")
print(f"    SMaRT: {t_smart_calib/60:.1f} min")
print(f"    RayFlare: {t_rf_calib/86400:.1f} days")

# Save results
import json
output = {
    'rayflare': {
        'tests': results,
        'avg_t_per_wl_s': avg_t_per_wl
    },
    'smart': {
        't_50wl_ms': t_smart * 1000,
        't_per_wl_ms': t_smart_per_wl * 1000
    },
    'speedup': speedup,
    'calibration': {
        'smart_min': t_smart_calib / 60,
        'rayflare_days': t_rf_calib / 86400
    }
}

output_file = PROJECT_ROOT / 'results' / 'benchmark_rayflare_real.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Results saved to: {output_file}")
