#!/usr/bin/env python3
"""
Example usage of Chronos MoE with both architectures.
Demonstrates how to configure and use shared_then_expert vs expert_only modes.
"""

import torch
import sys
sys.path.insert(0, 'src')

from chronos import ChronosConfig, ChronosMoEModel, ChronosMoEPipeline
from transformers import AutoModelForSeq2SeqLM


def create_moe_model(architecture="shared_then_expert", num_experts=8):
    """Create a MoE model with specified architecture."""
    
    print(f"Creating MoE model with {architecture} architecture...")
    
    # Configuration for MoE
    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={'low_limit': -10.0, 'high_limit': 10.0},
        n_tokens=1024,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        context_length=64,
        prediction_length=24,
        num_samples=10,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        
        # MoE Configuration
        use_moe=True,
        num_experts=num_experts,
        num_active_experts=2,  # Use top-2 experts
        load_balancing_weight=0.01,
        router_hidden_dim=256,
        moe_architecture=architecture,  # "shared_then_expert" or "expert_only"
    )
    
    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-efficient-tiny")
    base_model.resize_token_embeddings(config.n_tokens)
    
    # Create MoE model
    moe_model = ChronosMoEModel(config=config, model=base_model)
    
    # Create pipeline
    pipeline = ChronosMoEPipeline(
        tokenizer=config.create_tokenizer(),
        model=moe_model
    )
    
    return pipeline, config


def demonstrate_architectures():
    """Demonstrate both MoE architectures."""
    
    # Create sample time series data
    sample_ts = torch.randn(100)  # Random time series of length 100
    
    print("=" * 60)
    print("Chronos MoE Architecture Comparison")
    print("=" * 60)
    
    architectures = ["shared_then_expert", "expert_only"]
    
    for arch in architectures:
        print(f"\n--- {arch.upper()} ARCHITECTURE ---")
        
        # Create model
        pipeline, config = create_moe_model(architecture=arch, num_experts=4)
        
        # Count parameters
        total_params = sum(p.numel() for p in pipeline.model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Test routing behavior
        print("Testing routing behavior...")
        
        # Create test input
        batch_size = 3
        input_ids = torch.randint(0, config.n_tokens, (batch_size, config.context_length))
        attention_mask = torch.ones(batch_size, config.context_length, dtype=torch.bool)
        
        # Get routing information
        with torch.no_grad():
            shared_hidden = pipeline.model.get_shared_hidden_states(input_ids, attention_mask)
            router_logits, router_probs = pipeline.model.router(shared_hidden)
            expert_indices, top_k_probs = pipeline.model.router.get_top_k_experts(router_probs)
        
        print(f"Shared hidden shape: {shared_hidden.shape}")
        print(f"Expert assignments: {expert_indices.numpy()}")
        print(f"Expert probabilities: {top_k_probs.numpy()}")
        
        # Architecture-specific insights
        if arch == "shared_then_expert":
            print("✓ Shared encoder processes all inputs")
            print("✓ Experts specialize on prediction patterns")
            print("✓ More parameter efficient")
        else:  # expert_only
            print("✓ Experts process inputs from embedding layer")
            print("✓ Experts specialize on input AND prediction patterns")
            print("✓ Maximum specialization potential")


def compare_parameter_efficiency():
    """Compare parameter efficiency between architectures."""
    
    print("\n" + "=" * 60)
    print("PARAMETER EFFICIENCY COMPARISON")
    print("=" * 60)
    
    expert_counts = [4, 8, 16]
    
    for num_experts in expert_counts:
        print(f"\n--- {num_experts} Experts ---")
        
        for arch in ["shared_then_expert", "expert_only"]:
            pipeline, _ = create_moe_model(architecture=arch, num_experts=num_experts)
            total_params = sum(p.numel() for p in pipeline.model.parameters())
            print(f"{arch:20}: {total_params:,} parameters")
        
        # Calculate ratio
        shared_pipeline, _ = create_moe_model("shared_then_expert", num_experts)
        expert_pipeline, _ = create_moe_model("expert_only", num_experts)
        
        shared_params = sum(p.numel() for p in shared_pipeline.model.parameters())
        expert_params = sum(p.numel() for p in expert_pipeline.model.parameters())
        
        ratio = expert_params / shared_params
        print(f"{'Ratio (expert/shared)':20}: {ratio:.2f}x")


def usage_recommendations():
    """Provide usage recommendations for each architecture."""
    
    print("\n" + "=" * 60)
    print("USAGE RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n🔹 SHARED_THEN_EXPERT Architecture:")
    print("   Use when:")
    print("   • You want parameter efficiency")
    print("   • Experts should specialize on prediction patterns")
    print("   • You have limited computational resources")
    print("   • Input patterns are similar but prediction needs differ")
    print("   \n   Example: Different forecasting horizons or uncertainty levels")
    
    print("\n🔹 EXPERT_ONLY Architecture:")
    print("   Use when:")
    print("   • You want maximum expert specialization")
    print("   • Input patterns are fundamentally different")
    print("   • You have sufficient computational resources")
    print("   • You need interpretable expert behavior")
    print("   \n   Example: Different data domains (finance vs weather vs sales)")
    
    print("\n💡 Configuration Tips:")
    print("   • Start with shared_then_expert for most use cases")
    print("   • Use expert_only when you have diverse input patterns")
    print("   • Adjust num_experts based on data diversity")
    print("   • Use load_balancing_weight=0.01 to encourage expert utilization")


if __name__ == "__main__":
    print("🚀 Chronos MoE Architecture Demo")
    
    try:
        # Demonstrate both architectures
        demonstrate_architectures()
        
        # Compare parameter efficiency
        compare_parameter_efficiency()
        
        # Provide usage recommendations
        usage_recommendations()
        
        print("\n🎉 Demo completed successfully!")
        print("Choose the architecture that best fits your use case!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()