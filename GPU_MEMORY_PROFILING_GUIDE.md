# GPU Memory Profiling for ROCKET

Notes on using PyTorch's memory profiling tools to debug and optimize GPU memory usage in the new ROCKET refinement pipeline.

### Option 1: Memory Profiler (most cases)
Tracks overall memory usage patterns and categories.

```bash
python examples/memory_profiling_examples.py path/to/config.yaml --mode profile --output-dir ./memory_profiles
```
This generates:
- `hostname_timestamp_timeline.html` - Interactive timeline (open in browser)
- `hostname_timestamp_trace.json.gz` - Chrome trace (open in chrome://tracing)

### Option 2: Memory Snapshots (fordebugging OOMs)
Detailed allocation history with stack traces.

```bash
python examples/memory_profiling_examples.py path/to/config.yaml --mode snapshot --output-dir ./memory_profiles
```

Then upload the `.pickle` file to [https://pytorch.org/memory_viz](https://pytorch.org/memory_viz) to visualize.

## Two Profiling Approaches

### 1. Memory Profiler - Categorized Memory Timeline

**How it works:**
- Records memory allocations during execution
- Categorizes memory by operation type (forward, backward, optimizer)
- Shows memory usage over iteration

**Output breakdown:**
- Blue regions: Gradient tensors
- Yellow regions: Optimizer state
- Green regions: Activations (forward pass)
- Red regions: Buffers and temporary tensors

**Use in code:**
```python
from rocket.memory_profiler import MemoryProfiler

# Profile entire refinement run
with MemoryProfiler(output_dir, device="cuda:0"):
    run_panddamap_refinement(config_path)

# Or with manual step tracking for granular monitoring
profiler = MemoryProfiler(output_dir, device="cuda:0")
with profiler:
    # Your training code
    config = run_panddamap_refinement(config_path)
    profiler.step()  # Mark iteration end
```

### 2. Memory Snapshot - Detailed Allocation History

**How it works:**
- Records every GPU memory event (allocation/free) with stack trace
- Captures up to 100,000 recent events (configurable)
- Saves as pickle file for visualization
- Stack traces show exactly where allocations happen in code

**Advantages:**
- Full stack traces for debugging

**Use in your code:**
```python
from rocket.memory_profiler import MemorySnapshotProfiler

# Profile entire refinement run
with MemorySnapshotProfiler(output_dir):
    run_panddamap_refinement(config_path)
```


## Advanced Usage

### Decorator-based profiling

```python
from rocket.memory_profiler import memory_snapshot, memory_profile

@memory_snapshot(output_dir="./profiles")
def my_training_function(config):
    return run_panddamap_refinement(config)

# Or with memory profile
@memory_profile(output_dir="./profiles")
def my_training_function(config):
    return run_panddamap_refinement(config)
```

### Comparing multiple runs

```bash
# Run 1
python examples/memory_profiling_examples.py config1.yaml --mode snapshot --output-dir ./profiles_run1

# Run 2 with different settings
python examples/memory_profiling_examples.py config2.yaml --mode snapshot --output-dir ./profiles_run2

# Upload both .pickle files to https://pytorch.org/memory_viz for comparison
```
