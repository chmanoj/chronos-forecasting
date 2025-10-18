# MOE Model Parallelism Implementation Plan

## Executive Summary

This document outlines a comprehensive plan to implement model parallelism for Chronos Mixture of Experts (MOE) models, enabling efficient distribution of experts across multiple GPUs. The current implementation supports data parallelism but lacks true model parallelism where experts are distributed across different devices.

## Current State Analysis

### What Works Today ✅
- **Data Parallelism**: Full MOE model replicated across GPUs
- **MOE Architecture**: Well-structured with proper routing and load balancing
- **Device Awareness**: Proper device placement for single-GPU MOE
- **Training Infrastructure**: Hugging Face Trainer integration

### Current Limitations ❌
- All experts reside on the same GPU
- No cross-GPU expert communication
- Memory scales linearly with number of experts
- No expert-specific gradient optimization

## Implementation Plan

### Phase 1: Foundation (Week 1-2)

#### 1.1 Expert Distribution Strategy
**Goal**: Distribute experts across available GPUs

**Implementation**:
```python
class DistributedMoEModel(ChronosMoEModel):
    def __init__(self, config: ChronosConfig, model: PreTrainedModel):
        super().__init__(config, model)
        self.device_map = self._create_device_map()
        self._distribute_experts()
    
    def _create_device_map(self):
        """Create mapping of experts to devices."""
        num_gpus = torch.cuda.device_count()
        device_map = {}
        for i in range(self.config.num_experts):
            device_id = i % num_gpus
            device_map[i] = f'cuda:{device_id}'
        return device_map
    
    def _distribute_experts(self):
        """Move experts to their assigned devices."""
        for i, expert in enumerate(self.experts):
            target_device = self.device_map[i]
            expert.to(target_device)
```

**Files to Modify**:
- `src/chronos/chronos.py`: Add `DistributedMoEModel` class
- `scripts/training/train.py`: Add distributed MOE option

#### 1.2 Cross-GPU Communication Layer
**Goal**: Enable efficient tensor transfers between expert devices

**Implementation**:
```python
class ExpertCommunicator:
    def __init__(self, device_map, num_active_experts):
        self.device_map = device_map
        self.num_active_experts = num_active_experts
        
    def scatter_to_experts(self, hidden_states, expert_indices, router_probs):
        """Scatter hidden states to expert devices."""
        expert_inputs = {}
        for batch_idx in range(hidden_states.size(0)):
            for expert_idx in expert_indices[batch_idx]:
                expert_idx = expert_idx.item()
                target_device = self.device_map[expert_idx]
                
                if expert_idx not in expert_inputs:
                    expert_inputs[expert_idx] = []
                
                # Move sample to expert device
                sample_hidden = hidden_states[batch_idx:batch_idx+1].to(target_device)
                expert_inputs[expert_idx].append((batch_idx, sample_hidden))
        
        return expert_inputs
    
    def gather_from_experts(self, expert_outputs, batch_size, seq_len, vocab_size):
        """Gather expert outputs back to main device."""
        # Implementation for efficient gathering
        pass
```

#### 1.3 Memory-Efficient Routing
**Goal**: Optimize routing for cross-GPU scenarios

**Key Changes**:
- Router stays on main device
- Routing decisions cached to minimize transfers
- Batch-aware expert assignment

### Phase 2: Core Implementation (Week 3-4)

#### 2.1 Distributed Forward Pass
**Goal**: Implement efficient forward pass with expert distribution

**Architecture**:
```
Input (GPU 0) → Router (GPU 0) → Expert Assignment
                ↓
Expert 0 (GPU 0)    Expert 1 (GPU 1)    Expert 2 (GPU 2)    Expert 3 (GPU 3)
                ↓
Gather Results (GPU 0) → Combine Logits → Output
```

**Implementation Strategy**:
```python
def distributed_forward(self, input_ids, attention_mask, return_moe_loss=False):
    # 1. Get shared hidden states (main device)
    shared_hidden = self.get_shared_hidden_states(input_ids, attention_mask)
    
    # 2. Route samples to experts (main device)
    router_logits, router_probs = self.router(shared_hidden)
    expert_indices, top_k_probs = self.router.get_top_k_experts(router_probs)
    
    # 3. Scatter to expert devices
    expert_inputs = self.communicator.scatter_to_experts(
        shared_hidden, expert_indices, router_probs
    )
    
    # 4. Parallel expert processing
    expert_outputs = {}
    for expert_idx, inputs in expert_inputs.items():
        device = self.device_map[expert_idx]
        with torch.cuda.device(device):
            expert_outputs[expert_idx] = self.experts[expert_idx](inputs)
    
    # 5. Gather and combine results
    combined_logits = self.communicator.gather_from_experts(
        expert_outputs, batch_size, seq_len, vocab_size
    )
    
    return combined_logits
```

#### 2.2 Gradient Synchronization
**Goal**: Efficient gradient handling across expert devices

**Challenges**:
- Experts on different devices have different gradients
- Load balancing affects gradient distribution
- Need to synchronize router gradients

**Solution**:
```python
class DistributedMoETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_optimizers = self._create_expert_optimizers()
    
    def _create_expert_optimizers(self):
        """Create separate optimizers for each expert device."""
        optimizers = {}
        for expert_idx, device in self.model.device_map.items():
            expert_params = list(self.model.experts[expert_idx].parameters())
            optimizers[expert_idx] = torch.optim.AdamW(expert_params, lr=self.args.learning_rate)
        return optimizers
    
    def training_step(self, model, inputs):
        # Custom training step with distributed gradient handling
        pass
```

### Phase 3: Optimization (Week 5-6)

#### 3.1 Communication Optimization
**Goal**: Minimize cross-GPU communication overhead

**Strategies**:
1. **Batched Transfers**: Group multiple samples for same expert
2. **Asynchronous Communication**: Overlap computation and communication
3. **Compression**: Reduce tensor sizes where possible

**Implementation**:
```python
class OptimizedCommunicator(ExpertCommunicator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transfer_streams = self._create_cuda_streams()
    
    def _create_cuda_streams(self):
        """Create CUDA streams for async transfers."""
        streams = {}
        for device in set(self.device_map.values()):
            streams[device] = torch.cuda.Stream(device=device)
        return streams
    
    def async_scatter_to_experts(self, hidden_states, expert_indices):
        """Asynchronous scatter with CUDA streams."""
        # Implementation with async transfers
        pass
```

#### 3.2 Memory Management
**Goal**: Optimize memory usage across devices

**Features**:
1. **Expert Offloading**: Move unused experts to CPU
2. **Activation Checkpointing**: Reduce memory for large models
3. **Dynamic Expert Loading**: Load experts on-demand

**Implementation**:
```python
class MemoryOptimizedMoE(DistributedMoEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_cache = {}
        self.cpu_experts = {}
    
    def offload_unused_experts(self, active_experts):
        """Move unused experts to CPU memory."""
        for expert_idx in range(self.config.num_experts):
            if expert_idx not in active_experts:
                if expert_idx not in self.cpu_experts:
                    self.cpu_experts[expert_idx] = self.experts[expert_idx].cpu()
                # Clear GPU memory
                del self.experts[expert_idx]
    
    def load_expert_on_demand(self, expert_idx):
        """Load expert from CPU to GPU when needed."""
        if expert_idx in self.cpu_experts:
            device = self.device_map[expert_idx]
            self.experts[expert_idx] = self.cpu_experts[expert_idx].to(device)
```

### Phase 4: Advanced Features (Week 7-8)

#### 4.1 Dynamic Expert Scaling
**Goal**: Automatically adjust expert distribution based on load

**Features**:
- Monitor expert utilization
- Redistribute experts for load balancing
- Dynamic expert creation/removal

#### 4.2 Fault Tolerance
**Goal**: Handle GPU failures gracefully

**Features**:
- Expert redundancy across devices
- Automatic failover mechanisms
- Checkpoint/restore for expert states

#### 4.3 Performance Monitoring
**Goal**: Comprehensive monitoring and profiling

**Metrics**:
- Expert utilization per GPU
- Communication overhead
- Memory usage patterns
- Training throughput

## Implementation Details

### File Structure
```
src/chronos/
├── distributed/
│   ├── __init__.py
│   ├── distributed_moe.py          # DistributedMoEModel
│   ├── communication.py            # ExpertCommunicator
│   ├── memory_manager.py           # Memory optimization
│   └── trainer.py                  # DistributedMoETrainer
├── chronos.py                      # Updated with distributed imports
└── utils/
    ├── device_utils.py             # Device management utilities
    └── profiling.py                # Performance monitoring
```

### Configuration Changes
```python
@dataclass
class ChronosConfig:
    # ... existing fields ...
    
    # Model parallelism settings
    use_model_parallel: bool = False
    expert_parallel_strategy: str = "round_robin"  # "round_robin", "load_balanced", "custom"
    communication_backend: str = "nccl"
    enable_expert_offloading: bool = False
    max_experts_per_gpu: Optional[int] = None
    
    # Memory optimization
    use_activation_checkpointing: bool = False
    expert_cache_size: int = 2  # Number of experts to keep in GPU cache
    
    # Performance tuning
    async_communication: bool = True
    batch_expert_calls: bool = True
    communication_compression: bool = False
```

### Training Script Updates
```python
# scripts/training/train.py additions

def create_distributed_moe_model(config, base_model):
    """Create distributed MOE model based on config."""
    if config.use_model_parallel and torch.cuda.device_count() > 1:
        from chronos.distributed import DistributedMoEModel
        return DistributedMoEModel(config=config, model=base_model)
    else:
        return ChronosMoEModel(config=config, model=base_model)

def create_distributed_trainer(model, training_args, train_dataset, config):
    """Create appropriate trainer for distributed MOE."""
    if isinstance(model, DistributedMoEModel):
        from chronos.distributed import DistributedMoETrainer
        return DistributedMoETrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            config=config
        )
    else:
        return MoETrainer(...)  # Existing trainer
```

## Testing Strategy

### Unit Tests
1. **Expert Distribution**: Verify experts are on correct devices
2. **Communication**: Test scatter/gather operations
3. **Memory Management**: Validate offloading/loading
4. **Gradient Sync**: Ensure proper gradient handling

### Integration Tests
1. **Multi-GPU Training**: End-to-end training with multiple GPUs
2. **Performance Benchmarks**: Compare with data parallel baseline
3. **Memory Usage**: Monitor memory consumption patterns
4. **Fault Tolerance**: Test GPU failure scenarios

### Performance Tests
```python
# tests/test_distributed_moe.py
def test_multi_gpu_performance():
    """Compare distributed MOE vs data parallel performance."""
    # Setup both models
    # Run identical training steps
    # Compare throughput, memory usage, convergence
    pass

def test_expert_utilization():
    """Verify experts are being used efficiently."""
    # Monitor expert usage across GPUs
    # Check load balancing effectiveness
    pass
```

## Deployment Considerations

### Hardware Requirements
- **Minimum**: 2 GPUs with 8GB+ VRAM each
- **Recommended**: 4+ GPUs with 16GB+ VRAM each
- **Network**: High-bandwidth interconnect (NVLink preferred)

### Software Dependencies
```python
# requirements-distributed.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
accelerate>=0.20.0
deepspeed>=0.9.0  # Optional: for advanced optimizations
```

### Configuration Examples
```yaml
# configs/distributed-moe-config.yaml
use_moe: true
use_model_parallel: true
num_experts: 8
num_active_experts: 2
expert_parallel_strategy: "round_robin"
enable_expert_offloading: true
async_communication: true
```

## Migration Path

### Phase 1: Backward Compatibility
- Keep existing MOE implementation unchanged
- Add distributed MOE as optional feature
- Automatic fallback to data parallel if model parallel fails

### Phase 2: Gradual Adoption
- Default to model parallel for multi-GPU setups
- Provide clear migration guide
- Performance comparison tools

### Phase 3: Full Integration
- Deprecate old single-GPU MOE limitations
- Optimize for distributed-first approach
- Advanced features (dynamic scaling, etc.)

## Risk Mitigation

### Technical Risks
1. **Communication Overhead**: May slow down training
   - *Mitigation*: Extensive benchmarking, async communication
2. **Memory Fragmentation**: Complex memory patterns
   - *Mitigation*: Careful memory management, monitoring
3. **Debugging Complexity**: Harder to debug distributed issues
   - *Mitigation*: Comprehensive logging, visualization tools

### Implementation Risks
1. **Compatibility Issues**: May break existing workflows
   - *Mitigation*: Thorough testing, backward compatibility
2. **Performance Regression**: May be slower than data parallel
   - *Mitigation*: Performance gates, optimization focus

## Success Metrics

### Performance Metrics
- **Training Throughput**: Samples/second improvement
- **Memory Efficiency**: Memory usage per expert
- **Scaling Efficiency**: Performance vs number of GPUs
- **Communication Overhead**: Time spent in transfers

### Quality Metrics
- **Model Convergence**: Same convergence as baseline
- **Expert Utilization**: Balanced expert usage
- **Load Balancing**: Even GPU utilization

### Usability Metrics
- **Setup Complexity**: Time to configure distributed training
- **Debugging Experience**: Time to identify/fix issues
- **Documentation Quality**: User satisfaction scores

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1 | 2 weeks | Expert distribution, basic communication |
| Phase 2 | 2 weeks | Full distributed forward pass, gradient sync |
| Phase 3 | 2 weeks | Performance optimization, memory management |
| Phase 4 | 2 weeks | Advanced features, monitoring, documentation |

**Total Estimated Time**: 8 weeks for full implementation

## Conclusion

This plan provides a comprehensive roadmap for implementing model parallelism in the Chronos MOE system. The phased approach ensures incremental progress while maintaining system stability. The focus on performance optimization and thorough testing will ensure the distributed implementation meets production requirements.

The key to success will be careful attention to communication patterns, memory management, and maintaining the existing MOE training quality while achieving better scalability across multiple GPUs.