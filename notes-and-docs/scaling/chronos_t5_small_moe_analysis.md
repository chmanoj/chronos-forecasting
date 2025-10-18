# Chronos T5 Small MoE Parameter Analysis

## Base Model: Chronos T5 Small (46M Parameters)

### Architecture Configuration
- **Model Type**: T5 Encoder-Decoder
- **Encoder/Decoder Layers**: 6 each
- **Hidden Dimension**: 512
- **Feed-Forward Dimension**: 2048
- **Attention Heads**: 8
- **Vocabulary Size**: 4096 (reduced from T5's 32128)

### Parameter Breakdown by Component

| Component | Parameters | Details |
|-----------|------------|---------|
| **Token Embeddings** | 2.1M | 4096 × 512 |
| **Encoder (6 layers)** | 18.9M | |
| - Self-Attention | 6.3M | Q/K/V + output projections |
| - Feed-Forward | 12.6M | 512 → 2048 → 512 per layer |
| **Decoder (6 layers)** | 25.2M | |
| - Self-Attention | 6.3M | Same as encoder |
| - Cross-Attention | 6.3M | Encoder-decoder attention |
| - Feed-Forward | 12.6M | Same structure as encoder |
| **Output Head** | 2.1M | 512 × 4096 final projection |
| **Total** | **46M** | |

## MoE Transformation: 8 Experts, 2 Active

### MoE Architecture Design
- **Shared Components**: Encoder + most decoder layers
- **Expert Specialization**: Replace final prediction layers with expert heads
- **Routing Strategy**: Context-based sample-level routing (not token-level)

### Parameter Changes

#### Shared Components (Unchanged): 44M
- Token embeddings: 2.1M
- Encoder (6 layers): 18.9M
- Decoder shared layers: ~23M
- Layer norms and misc: ~0.1M

#### MoE-Specific Components: 34M
- **Context Router**: 0.4M parameters
  - Input projection: 512 → 256
  - Output projection: 256 → 8 experts
- **8 Expert Heads**: 33.6M parameters
  - Per expert: 4.2M (FFN + LayerNorm + LM head)
  - Total: 8 × 4.2M = 33.6M

## Parameter Comparison

| Metric | Base Model | MoE Model | Change |
|--------|------------|-----------|---------|
| **Total Parameters** | 46M | 78M | +70% |
| **Active Parameters** | 46M | 53M | +15% |
| **Storage Required** | 184MB | 312MB | +70% |
| **Inference Compute** | 46M ops | 53M ops | +15% |

## Efficiency Analysis

### Parameter Efficiency
- **Total Capacity**: 78M parameters
- **Active Usage**: 53M parameters  
- **Efficiency Ratio**: 1.47× (78M/53M)
- **Expert Utilization**: 25% (2/8 experts active)

### Computational Benefits
- **Specialization**: 8× model capacity for different forecasting patterns
- **Compute Overhead**: Only 15% increase in active parameters
- **Memory Efficiency**: Load balancing prevents expert underutilization
- **Inference Speed**: Minimal impact due to shared encoder/decoder

### Trade-offs
**Advantages:**
- 8× specialized forecasting capacity
- Minimal compute increase (15%)
- Better handling of diverse time series patterns
- Scalable expert architecture

**Disadvantages:**
- 70% increase in model storage
- Added routing complexity
- Load balancing overhead
- More complex training dynamics

## Recommended Use Cases

**Ideal for MoE:**
- Diverse time series datasets (multiple domains/patterns)
- Production environments with varied forecasting needs
- When model specialization outweighs storage costs

**Stick with Base Model:**
- Homogeneous time series patterns
- Storage/memory constrained environments
- Simple deployment requirements
- When 46M parameters provide sufficient capacity

## Implementation Notes

Based on the codebase architecture:
- Uses `ChronosMoEModel` with `ContextRouter`
- Sample-level routing (not token-level)
- Expert heads include FFN + prediction layers
- Load balancing loss prevents expert collapse
- Compatible with existing Chronos tokenization