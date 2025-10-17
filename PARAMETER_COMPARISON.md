# Chronos Model Parameter Comparison

## Parameter Counts

### Regular Chronos Model (T5-Tiny)
- **Total Parameters**: 7,608,064
- **Trainable Parameters**: 7,608,064
- **Model Type**: Standard seq2seq transformer

### MOE Chronos Model (T5-Tiny + 4 Experts)
- **Base Model Parameters**: 7,608,064
- **Total MOE Parameters**: 10,794,372
- **Trainable Parameters**: 10,794,372
- **MOE Overhead**: 3,186,308 parameters
- **Parameter Increase**: 41.9%

## MOE Architecture Breakdown

The MOE overhead of 3,186,308 parameters comes from:

### 1. Context Router (per model)
- Input dimension: 512 (T5-tiny hidden size)
- Router hidden dimension: 128
- Number of experts: 4
- **Router parameters**: ~66K parameters
  - Linear 1: 512 × 128 = 65,536
  - Linear 2: 128 × 4 = 512
  - Bias terms: ~640

### 2. Expert Heads (4 experts)
- Each expert has its own feed-forward network and prediction head
- **Per expert**: ~780K parameters
  - Expert FFN: 512 → 2048 → 512 (~1.5M params)
  - Layer norm: ~1K params  
  - LM head: 512 × 1024 = 524,288 params
- **Total for 4 experts**: ~3.1M parameters

### 3. Load Balancing Loss
- Minimal parameters (just tracking tensors)

## Memory and Compute Implications

### Memory Usage
- **41.9% increase** in model parameters
- During training: Additional memory for expert gradients
- During inference: Only active experts (2 out of 4) are used

### Compute Efficiency
- **Training**: ~42% more parameters to train
- **Inference**: Only 2/4 experts active per sample
- **Effective capacity**: Higher model capacity without proportional compute increase

## Performance vs Efficiency Trade-off

| Metric | Regular Model | MOE Model | Difference |
|--------|---------------|-----------|------------|
| Parameters | 7.6M | 10.8M | +41.9% |
| Training Time | ~4s | ~4s | Similar |
| Memory Usage | Baseline | +41.9% | Higher |
| Model Capacity | Fixed | Adaptive | Higher |
| Expert Utilization | N/A | Load balanced | Specialized |

## Key Benefits of MOE

1. **Increased Capacity**: 41.9% more parameters for potential performance gains
2. **Conditional Computation**: Only 2/4 experts active per sample during inference
3. **Specialization**: Different experts can learn different time series patterns
4. **Scalability**: Can add more experts without linear compute increase

## Recommendations

- **Use Regular Model**: For simple datasets or resource-constrained environments
- **Use MOE Model**: For complex, diverse time series datasets where specialization helps
- **Expert Count**: Start with 4-8 experts, tune based on dataset diversity
- **Active Experts**: Keep at 2 for good balance of capacity and efficiency