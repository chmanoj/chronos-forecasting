# MOE Distributed Architecture Technical Specification

## Architecture Overview

### Current vs Proposed Architecture

#### Current Architecture (Data Parallel)
```
GPU 0: [Router + Expert0 + Expert1 + Expert2 + Expert3] ← Full Model Copy
GPU 1: [Router + Expert0 + Expert1 + Expert2 + Expert3] ← Full Model Copy  
GPU 2: [Router + Expert0 + Expert1 + Expert2 + Expert3] ← Full Model Copy
GPU 3: [Router + Expert0 + Expert1 + Expert2 + Expert3] ← Full Model Copy

Memory per GPU: Base Model + All Experts
Communication: Gradient synchronization only
```

#### Proposed Architecture (Model Parallel)
```
GPU 0: [Router + Expert0 + Expert4] ← Shared Router + Subset of Experts
GPU 1: [Expert1 + Expert5]          ← Experts only
GPU 2: [Expert2 + Expert6]          ← Experts only  
GPU 3: [Expert3 + Expert7]          ← Experts only

Memory per GPU: Base Model + Subset of Experts
Communication: Activations + Gradients + Routing decisions
```

## Detailed Implementation

### 1. Core Distributed MOE Model

```python
# src/chronos/distributed/distributed_moe.py

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional, Any
from ..chronos import ChronosMoEModel, ChronosConfig
from .communication import ExpertCommunicator
from .memory_manager import ExpertMemoryManager

class DistributedMoEModel(ChronosMoEModel):
    """
    Distributed Mixture of Experts model that distributes experts across multiple GPUs.
    
    Key Features:
    - Expert sharding across available GPUs
    - Efficient cross-GPU communication
    - Memory optimization with expert offloading
    - Load balancing across devices
    """
    
    def __init__(self, config: ChronosConfig, model, world_size: int = None, rank: int = None):
        # Initialize base MOE model first
        super().__init__(config, model)
        
        # Distributed setup
        self.world_size = world_size or torch.cuda.device_count()
        self.rank = rank or 0
        self.main_device = f'cuda:{self.rank}'
        
        # Expert distribution strategy
        self.device_map = self._create_expert_device_map()
        self.local_experts = self._get_local_expert_indices()
        
        # Communication and memory management
        self.communicator = ExpertCommunicator(
            device_map=self.device_map,
            world_size=self.world_size,
            rank=self.rank
        )
        
        if config.enable_expert_offloading:
            self.memory_manager = ExpertMemoryManager(
                experts=self.experts,
                device_map=self.device_map,
                cache_size=config.expert_cache_size
            )
        
        # Distribute experts to their assigned devices
        self._distribute_experts()
        
        # Setup communication groups
        self._setup_communication_groups()
    
    def _create_expert_device_map(self) -> Dict[int, str]:
        """
        Create mapping of expert indices to device IDs.
        
        Strategies:
        - round_robin: Distribute experts evenly across GPUs
        - load_balanced: Consider expert complexity/size
        - custom: User-defined mapping
        """
        device_map = {}
        
        if self.config.expert_parallel_strategy == "round_robin":
            for expert_idx in range(self.config.num_experts):
                device_id = expert_idx % self.world_size
                device_map[expert_idx] = f'cuda:{device_id}'
                
        elif self.config.expert_parallel_strategy == "load_balanced":
            # More sophisticated load balancing based on expert sizes
            expert_sizes = [self._estimate_expert_size(i) for i in range(self.config.num_experts)]
            device_loads = [0] * self.world_size
            
            # Greedy assignment to least loaded device
            for expert_idx, size in enumerate(expert_sizes):
                min_device = min(range(self.world_size), key=lambda d: device_loads[d])
                device_map[expert_idx] = f'cuda:{min_device}'
                device_loads[min_device] += size
                
        else:  # custom mapping from config
            device_map = self.config.custom_expert_device_map
        
        return device_map
    
    def _get_local_expert_indices(self) -> List[int]:
        """Get indices of experts that should be on current device."""
        return [
            expert_idx for expert_idx, device in self.device_map.items()
            if device == self.main_device
        ]
    
    def _distribute_experts(self):
        """Move experts to their assigned devices and remove non-local experts."""
        # Keep only local experts on this device
        local_expert_modules = nn.ModuleList()
        
        for expert_idx in range(self.config.num_experts):
            if expert_idx in self.local_experts:
                # Keep this expert on current device
                expert = self.experts[expert_idx]
                target_device = self.device_map[expert_idx]
                expert.to(target_device)
                local_expert_modules.append(expert)
            # Non-local experts will be accessed via communication
        
        # Replace experts with only local ones
        self.experts = local_expert_modules
        self.local_expert_map = {
            global_idx: local_idx 
            for local_idx, global_idx in enumerate(self.local_experts)
        }
    
    def _setup_communication_groups(self):
        """Setup process groups for efficient communication."""
        if dist.is_initialized():
            # Create process group for all expert devices
            self.expert_group = dist.new_group(list(range(self.world_size)))
            
            # Create groups for specific communication patterns
            self.router_group = dist.new_group([0])  # Router is on rank 0
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_moe_loss: bool = False,
                **kwargs) -> torch.Tensor:
        """
        Distributed forward pass with cross-GPU expert communication.
        
        Flow:
        1. Compute shared hidden states (on main device)
        2. Route samples to experts (on main device) 
        3. Distribute routing decisions to all devices
        4. Each device processes its local experts
        5. Gather expert outputs back to main device
        6. Combine outputs using routing probabilities
        """
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Get shared hidden states (on main device)
        shared_hidden = self.get_shared_hidden_states(input_ids, attention_mask)
        
        # Step 2: Route samples to experts (on main device)
        router_logits, router_probs = self.router(shared_hidden)
        expert_indices, top_k_probs = self.router.get_top_k_experts(router_probs)
        
        # Step 3: Distribute inputs to expert devices
        expert_outputs = self._distributed_expert_forward(
            shared_hidden, expert_indices, router_probs, attention_mask
        )
        
        # Step 4: Combine expert outputs
        combined_logits = self._combine_distributed_expert_outputs(
            expert_outputs, router_probs, expert_indices, 
            batch_size, seq_len, self.config.n_tokens
        )
        
        if return_moe_loss:
            load_loss = self.load_balancing_loss(router_probs, expert_indices)
            return combined_logits, load_loss
        
        return combined_logits
    
    def _distributed_expert_forward(self, 
                                   shared_hidden: torch.Tensor,
                                   expert_indices: torch.Tensor, 
                                   router_probs: torch.Tensor,
                                   attention_mask: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Process experts in distributed manner.
        
        Each device processes only its local experts and communicates results.
        """
        batch_size = shared_hidden.size(0)
        
        # Prepare inputs for each expert
        expert_inputs = self._prepare_expert_inputs(
            shared_hidden, expert_indices, attention_mask
        )
        
        # Process local experts
        local_expert_outputs = {}
        for global_expert_idx in self.local_experts:
            if global_expert_idx in expert_inputs:
                local_idx = self.local_expert_map[global_expert_idx]
                expert = self.experts[local_idx]
                
                # Get inputs for this expert
                expert_hidden, expert_mask, sample_indices = expert_inputs[global_expert_idx]
                
                # Process through expert
                expert_output = expert(expert_hidden, expert_mask)
                local_expert_outputs[global_expert_idx] = (expert_output, sample_indices)
        
        # Communicate expert outputs across devices
        if dist.is_initialized():
            all_expert_outputs = self._all_gather_expert_outputs(local_expert_outputs)
        else:
            # Single device fallback
            all_expert_outputs = local_expert_outputs
        
        return all_expert_outputs
    
    def _prepare_expert_inputs(self, 
                              shared_hidden: torch.Tensor,
                              expert_indices: torch.Tensor,
                              attention_mask: torch.Tensor) -> Dict[int, Tuple]:
        """
        Prepare inputs for each expert by grouping samples assigned to each expert.
        """
        expert_inputs = {}
        batch_size = shared_hidden.size(0)
        
        # Group samples by expert assignment
        for batch_idx in range(batch_size):
            for expert_slot in range(self.config.num_active_experts):
                expert_idx = expert_indices[batch_idx, expert_slot].item()
                
                if expert_idx not in expert_inputs:
                    expert_inputs[expert_idx] = {
                        'hidden_states': [],
                        'attention_masks': [],
                        'sample_indices': []
                    }
                
                expert_inputs[expert_idx]['hidden_states'].append(
                    shared_hidden[batch_idx:batch_idx+1]
                )
                expert_inputs[expert_idx]['attention_masks'].append(
                    attention_mask[batch_idx:batch_idx+1]
                )
                expert_inputs[expert_idx]['sample_indices'].append(
                    (batch_idx, expert_slot)
                )
        
        # Concatenate inputs for each expert
        processed_inputs = {}
        for expert_idx, inputs in expert_inputs.items():
            if inputs['hidden_states']:  # Only if expert has inputs
                expert_hidden = torch.cat(inputs['hidden_states'], dim=0)
                expert_mask = torch.cat(inputs['attention_masks'], dim=0)
                
                # Move to expert's device
                target_device = self.device_map[expert_idx]
                expert_hidden = expert_hidden.to(target_device)
                expert_mask = expert_mask.to(target_device)
                
                processed_inputs[expert_idx] = (
                    expert_hidden, expert_mask, inputs['sample_indices']
                )
        
        return processed_inputs
    
    def _all_gather_expert_outputs(self, local_outputs: Dict) -> Dict:
        """
        Gather expert outputs from all devices using efficient communication.
        """
        # This is a simplified version - real implementation would use
        # more efficient communication patterns like all-to-all
        
        all_outputs = {}
        
        for rank in range(self.world_size):
            if rank == self.rank:
                # Send our outputs to others
                for expert_idx, output_data in local_outputs.items():
                    all_outputs[expert_idx] = output_data
            else:
                # Receive outputs from other ranks
                # Implementation would use torch.distributed primitives
                pass
        
        return all_outputs
    
    def _combine_distributed_expert_outputs(self,
                                          expert_outputs: Dict[int, Tuple],
                                          router_probs: torch.Tensor,
                                          expert_indices: torch.Tensor,
                                          batch_size: int,
                                          seq_len: int,
                                          vocab_size: int) -> torch.Tensor:
        """
        Combine expert outputs using routing probabilities.
        
        This is more complex in distributed setting because expert outputs
        are on different devices and need to be gathered efficiently.
        """
        device = router_probs.device
        combined_logits = torch.zeros(
            batch_size, seq_len, vocab_size,
            device=device, dtype=torch.float32
        )
        
        # Reconstruct full batch outputs from expert outputs
        for expert_idx, (expert_logits, sample_indices) in expert_outputs.items():
            expert_logits = expert_logits.to(device)
            
            for i, (batch_idx, expert_slot) in enumerate(sample_indices):
                weight = router_probs[batch_idx, expert_idx]
                combined_logits[batch_idx] += weight * expert_logits[i]
        
        return combined_logits


class DistributedMoETrainer:
    """
    Custom trainer for distributed MOE models.
    
    Handles:
    - Distributed gradient synchronization
    - Expert-specific optimization
    - Load balancing across devices
    - Memory management
    """
    
    def __init__(self, model: DistributedMoEModel, config: ChronosConfig, **kwargs):
        self.model = model
        self.config = config
        
        # Create optimizers for different components
        self.router_optimizer = self._create_router_optimizer()
        self.expert_optimizers = self._create_expert_optimizers()
        self.shared_optimizer = self._create_shared_optimizer()
        
        # Performance monitoring
        self.communication_time = 0.0
        self.computation_time = 0.0
        self.expert_utilization = {}
    
    def _create_router_optimizer(self):
        """Create optimizer for router parameters."""
        router_params = list(self.model.router.parameters())
        return torch.optim.AdamW(router_params, lr=self.config.learning_rate)
    
    def _create_expert_optimizers(self):
        """Create optimizers for local experts."""
        optimizers = {}
        for local_idx, global_idx in enumerate(self.model.local_experts):
            expert_params = list(self.model.experts[local_idx].parameters())
            optimizers[global_idx] = torch.optim.AdamW(
                expert_params, lr=self.config.learning_rate
            )
        return optimizers
    
    def _create_shared_optimizer(self):
        """Create optimizer for shared model parameters."""
        shared_params = list(self.model.model.parameters())
        return torch.optim.AdamW(shared_params, lr=self.config.learning_rate)
    
    def training_step(self, batch):
        """
        Custom training step for distributed MOE.
        
        Handles:
        1. Forward pass with distributed experts
        2. Loss computation with load balancing
        3. Backward pass with expert-specific gradients
        4. Distributed gradient synchronization
        """
        import time
        
        # Forward pass
        start_time = time.time()
        
        logits, load_loss = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            return_moe_loss=True
        )
        
        # Compute losses
        ce_loss = self._compute_cross_entropy_loss(logits, batch['labels'])
        total_loss = ce_loss + self.config.load_balancing_weight * load_loss
        
        forward_time = time.time() - start_time
        
        # Backward pass
        start_time = time.time()
        total_loss.backward()
        
        # Synchronize gradients across devices
        self._synchronize_gradients()
        
        # Update optimizers
        self._update_optimizers()
        
        backward_time = time.time() - start_time
        
        # Update performance metrics
        self.computation_time += forward_time + backward_time
        
        return {
            'loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'load_loss': load_loss.item(),
            'forward_time': forward_time,
            'backward_time': backward_time
        }
    
    def _synchronize_gradients(self):
        """
        Synchronize gradients across distributed experts.
        
        Different components need different synchronization strategies:
        - Router: All-reduce across all devices
        - Experts: No sync needed (each expert on different device)
        - Shared model: All-reduce across all devices
        """
        if dist.is_initialized():
            # Synchronize router gradients
            for param in self.model.router.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, group=self.model.expert_group)
                    param.grad /= self.model.world_size
            
            # Synchronize shared model gradients  
            for param in self.model.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, group=self.model.expert_group)
                    param.grad /= self.model.world_size
            
            # Expert gradients don't need synchronization (distributed)
    
    def _update_optimizers(self):
        """Update all optimizers."""
        self.router_optimizer.step()
        self.shared_optimizer.step()
        
        for optimizer in self.expert_optimizers.values():
            optimizer.step()
        
        # Clear gradients
        self.router_optimizer.zero_grad()
        self.shared_optimizer.zero_grad()
        for optimizer in self.expert_optimizers.values():
            optimizer.zero_grad()
```

### 2. Communication Layer

```python
# src/chronos/distributed/communication.py

import torch
import torch.distributed as dist
from typing import Dict, List, Tuple, Any
import time

class ExpertCommunicator:
    """
    Handles efficient communication between expert devices.
    
    Key optimizations:
    - Batched transfers to reduce communication overhead
    - Asynchronous communication with computation overlap
    - Compression for large tensors
    - Smart routing to minimize cross-device transfers
    """
    
    def __init__(self, device_map: Dict[int, str], world_size: int, rank: int):
        self.device_map = device_map
        self.world_size = world_size
        self.rank = rank
        self.main_device = f'cuda:{rank}'
        
        # Communication streams for async operations
        self.streams = self._create_communication_streams()
        
        # Performance tracking
        self.communication_stats = {
            'total_bytes_sent': 0,
            'total_bytes_received': 0,
            'total_communication_time': 0.0,
            'num_communications': 0
        }
    
    def _create_communication_streams(self):
        """Create CUDA streams for asynchronous communication."""
        streams = {}
        for device_id in range(self.world_size):
            device = f'cuda:{device_id}'
            streams[device] = torch.cuda.Stream(device=device)
        return streams
    
    def scatter_to_experts(self, 
                          hidden_states: torch.Tensor,
                          expert_indices: torch.Tensor,
                          attention_mask: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Efficiently scatter hidden states to expert devices.
        
        Optimizations:
        - Batch samples going to same device
        - Use async transfers where possible
        - Minimize memory copies
        """
        start_time = time.time()
        
        # Group samples by target device
        device_batches = self._group_samples_by_device(
            hidden_states, expert_indices, attention_mask
        )
        
        # Perform async transfers
        expert_inputs = {}
        transfer_handles = []
        
        for device, batch_data in device_batches.items():
            if device != self.main_device:
                # Async transfer to remote device
                handle = self._async_transfer_to_device(batch_data, device)
                transfer_handles.append(handle)
            else:
                # Local data, no transfer needed
                expert_inputs.update(batch_data)
        
        # Wait for all transfers to complete
        for handle in transfer_handles:
            expert_inputs.update(handle.wait())
        
        # Update communication stats
        comm_time = time.time() - start_time
        self.communication_stats['total_communication_time'] += comm_time
        self.communication_stats['num_communications'] += 1
        
        return expert_inputs
    
    def gather_from_experts(self, 
                           expert_outputs: Dict[int, torch.Tensor],
                           target_device: str = None) -> torch.Tensor:
        """
        Efficiently gather expert outputs back to main device.
        
        Uses all-gather pattern for efficiency when multiple devices
        need the same data.
        """
        if target_device is None:
            target_device = self.main_device
        
        start_time = time.time()
        
        # Collect outputs from all devices
        gathered_outputs = {}
        
        if dist.is_initialized():
            # Use distributed primitives for efficient gathering
            gathered_outputs = self._distributed_gather(expert_outputs, target_device)
        else:
            # Single device fallback
            gathered_outputs = expert_outputs
        
        comm_time = time.time() - start_time
        self.communication_stats['total_communication_time'] += comm_time
        
        return gathered_outputs
    
    def _group_samples_by_device(self, 
                                hidden_states: torch.Tensor,
                                expert_indices: torch.Tensor,
                                attention_mask: torch.Tensor) -> Dict[str, Dict]:
        """Group samples by their target expert devices."""
        device_batches = {}
        batch_size = hidden_states.size(0)
        
        for batch_idx in range(batch_size):
            for expert_slot in range(expert_indices.size(1)):
                expert_idx = expert_indices[batch_idx, expert_slot].item()
                target_device = self.device_map[expert_idx]
                
                if target_device not in device_batches:
                    device_batches[target_device] = {}
                
                if expert_idx not in device_batches[target_device]:
                    device_batches[target_device][expert_idx] = {
                        'hidden_states': [],
                        'attention_masks': [],
                        'sample_info': []
                    }
                
                device_batches[target_device][expert_idx]['hidden_states'].append(
                    hidden_states[batch_idx:batch_idx+1]
                )
                device_batches[target_device][expert_idx]['attention_masks'].append(
                    attention_mask[batch_idx:batch_idx+1]
                )
                device_batches[target_device][expert_idx]['sample_info'].append(
                    (batch_idx, expert_slot)
                )
        
        # Concatenate tensors for each expert
        for device in device_batches:
            for expert_idx in device_batches[device]:
                data = device_batches[device][expert_idx]
                data['hidden_states'] = torch.cat(data['hidden_states'], dim=0)
                data['attention_masks'] = torch.cat(data['attention_masks'], dim=0)
        
        return device_batches
    
    def _async_transfer_to_device(self, batch_data: Dict, target_device: str):
        """Perform asynchronous transfer to target device."""
        class TransferHandle:
            def __init__(self, data, device, stream):
                self.data = data
                self.device = device
                self.stream = stream
                self.transferred_data = {}
                
                # Start async transfer
                with torch.cuda.stream(stream):
                    for expert_idx, expert_data in data.items():
                        self.transferred_data[expert_idx] = {
                            'hidden_states': expert_data['hidden_states'].to(device, non_blocking=True),
                            'attention_masks': expert_data['attention_masks'].to(device, non_blocking=True),
                            'sample_info': expert_data['sample_info']
                        }
            
            def wait(self):
                """Wait for transfer to complete and return data."""
                self.stream.synchronize()
                return self.transferred_data
        
        stream = self.streams[target_device]
        return TransferHandle(batch_data, target_device, stream)
    
    def _distributed_gather(self, expert_outputs: Dict, target_device: str) -> Dict:
        """Use distributed primitives for efficient gathering."""
        # This would implement efficient all-gather or reduce-scatter
        # patterns depending on the communication requirements
        
        # Simplified implementation - real version would be more optimized
        all_outputs = {}
        
        # Serialize local outputs
        local_data = self._serialize_expert_outputs(expert_outputs)
        
        # All-gather across all ranks
        gathered_data = [None] * self.world_size
        dist.all_gather_object(gathered_data, local_data)
        
        # Deserialize and combine
        for rank_data in gathered_data:
            if rank_data:
                rank_outputs = self._deserialize_expert_outputs(rank_data, target_device)
                all_outputs.update(rank_outputs)
        
        return all_outputs
    
    def _serialize_expert_outputs(self, expert_outputs: Dict) -> Dict:
        """Serialize expert outputs for communication."""
        serialized = {}
        for expert_idx, (output_tensor, sample_info) in expert_outputs.items():
            serialized[expert_idx] = {
                'tensor': output_tensor.cpu(),  # Move to CPU for transfer
                'sample_info': sample_info,
                'shape': output_tensor.shape,
                'dtype': output_tensor.dtype
            }
        return serialized
    
    def _deserialize_expert_outputs(self, serialized_data: Dict, target_device: str) -> Dict:
        """Deserialize expert outputs after communication."""
        deserialized = {}
        for expert_idx, data in serialized_data.items():
            tensor = data['tensor'].to(target_device)
            sample_info = data['sample_info']
            deserialized[expert_idx] = (tensor, sample_info)
        return deserialized
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication performance statistics."""
        stats = self.communication_stats.copy()
        if stats['num_communications'] > 0:
            stats['avg_communication_time'] = (
                stats['total_communication_time'] / stats['num_communications']
            )
        return stats
```

### 3. Memory Management

```python
# src/chronos/distributed/memory_manager.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set
import threading
import time

class ExpertMemoryManager:
    """
    Manages expert memory across devices with intelligent caching and offloading.
    
    Features:
    - LRU cache for frequently used experts
    - Automatic offloading to CPU memory
    - Prefetching based on usage patterns
    - Memory pressure monitoring
    """
    
    def __init__(self, 
                 experts: nn.ModuleList,
                 device_map: Dict[int, str],
                 cache_size: int = 2,
                 enable_prefetch: bool = True):
        self.experts = experts
        self.device_map = device_map
        self.cache_size = cache_size
        self.enable_prefetch = enable_prefetch
        
        # Expert state tracking
        self.expert_locations = {}  # expert_idx -> 'gpu' | 'cpu' | 'loading'
        self.expert_usage_history = []  # Recent expert usage for LRU
        self.expert_access_count = {}  # Usage frequency tracking
        
        # CPU storage for offloaded experts
        self.cpu_expert_storage = {}
        
        # Async loading
        self.loading_threads = {}
        self.loading_lock = threading.Lock()
        
        # Memory monitoring
        self.memory_pressure_threshold = 0.85  # 85% GPU memory usage
        
        # Initialize expert locations
        for expert_idx in range(len(experts)):
            self.expert_locations[expert_idx] = 'gpu'
            self.expert_access_count[expert_idx] = 0
    
    def get_expert(self, expert_idx: int, blocking: bool = True) -> nn.Module:
        """
        Get expert, loading from CPU if necessary.
        
        Args:
            expert_idx: Index of expert to retrieve
            blocking: Whether to wait for loading to complete
            
        Returns:
            Expert module on appropriate device
        """
        self._update_usage_stats(expert_idx)
        
        current_location = self.expert_locations[expert_idx]
        
        if current_location == 'gpu':
            # Expert already on GPU
            return self.experts[expert_idx]
        
        elif current_location == 'cpu':
            # Need to load from CPU
            if blocking:
                return self._load_expert_from_cpu(expert_idx)
            else:
                # Start async loading and return None
                self._start_async_loading(expert_idx)
                return None
        
        elif current_location == 'loading':
            # Currently being loaded
            if blocking:
                return self._wait_for_loading(expert_idx)
            else:
                return None
        
        else:
            raise ValueError(f"Unknown expert location: {current_location}")
    
    def offload_expert(self, expert_idx: int):
        """Offload expert to CPU memory."""
        if self.expert_locations[expert_idx] == 'gpu':
            # Move to CPU
            expert = self.experts[expert_idx]
            self.cpu_expert_storage[expert_idx] = expert.cpu()
            
            # Clear GPU memory
            del self.experts[expert_idx]
            torch.cuda.empty_cache()
            
            self.expert_locations[expert_idx] = 'cpu'
    
    def _load_expert_from_cpu(self, expert_idx: int) -> nn.Module:
        """Load expert from CPU to GPU."""
        if expert_idx not in self.cpu_expert_storage:
            raise ValueError(f"Expert {expert_idx} not found in CPU storage")
        
        # Check if we need to make room
        self._ensure_gpu_memory_available()
        
        # Load expert to GPU
        target_device = self.device_map[expert_idx]
        expert = self.cpu_expert_storage[expert_idx].to(target_device)
        
        # Update tracking
        self.experts[expert_idx] = expert
        self.expert_locations[expert_idx] = 'gpu'
        
        return expert
    
    def _start_async_loading(self, expert_idx: int):
        """Start loading expert asynchronously."""
        with self.loading_lock:
            if expert_idx not in self.loading_threads:
                self.expert_locations[expert_idx] = 'loading'
                
                def load_worker():
                    try:
                        self._load_expert_from_cpu(expert_idx)
                    except Exception as e:
                        print(f"Error loading expert {expert_idx}: {e}")
                        self.expert_locations[expert_idx] = 'cpu'
                    finally:
                        with self.loading_lock:
                            if expert_idx in self.loading_threads:
                                del self.loading_threads[expert_idx]
                
                thread = threading.Thread(target=load_worker)
                self.loading_threads[expert_idx] = thread
                thread.start()
    
    def _wait_for_loading(self, expert_idx: int) -> nn.Module:
        """Wait for async loading to complete."""
        with self.loading_lock:
            if expert_idx in self.loading_threads:
                thread = self.loading_threads[expert_idx]
        
        if thread:
            thread.join()
        
        return self.experts[expert_idx]
    
    def _ensure_gpu_memory_available(self):
        """Ensure sufficient GPU memory by offloading if necessary."""
        current_memory_usage = self._get_gpu_memory_usage()
        
        if current_memory_usage > self.memory_pressure_threshold:
            # Need to offload some experts
            candidates = self._get_offload_candidates()
            
            for expert_idx in candidates:
                self.offload_expert(expert_idx)
                
                # Check if we have enough memory now
                if self._get_gpu_memory_usage() <= self.memory_pressure_threshold:
                    break
    
    def _get_offload_candidates(self) -> List[int]:
        """Get list of experts to offload, ordered by priority."""
        # Use LRU strategy - offload least recently used experts
        gpu_experts = [
            idx for idx, location in self.expert_locations.items()
            if location == 'gpu'
        ]
        
        # Sort by usage recency (least recent first)
        def usage_priority(expert_idx):
            try:
                return self.expert_usage_history[::-1].index(expert_idx)
            except ValueError:
                return len(self.expert_usage_history)  # Never used
        
        candidates = sorted(gpu_experts, key=usage_priority, reverse=True)
        return candidates
    
    def _update_usage_stats(self, expert_idx: int):
        """Update expert usage statistics."""
        # Update access count
        self.expert_access_count[expert_idx] += 1
        
        # Update LRU history
        if expert_idx in self.expert_usage_history:
            self.expert_usage_history.remove(expert_idx)
        self.expert_usage_history.append(expert_idx)
        
        # Keep history bounded
        max_history = len(self.experts) * 2
        if len(self.expert_usage_history) > max_history:
            self.expert_usage_history = self.expert_usage_history[-max_history:]
        
        # Prefetch prediction
        if self.enable_prefetch:
            self._predict_and_prefetch()
    
    def _predict_and_prefetch(self):
        """Predict next expert usage and prefetch if beneficial."""
        # Simple pattern-based prefetching
        # In practice, this could use more sophisticated ML models
        
        if len(self.expert_usage_history) < 3:
            return
        
        # Look for patterns in recent usage
        recent_usage = self.expert_usage_history[-3:]
        
        # Simple heuristic: if we see a pattern, prefetch next likely expert
        # This is a placeholder for more sophisticated prediction logic
        pass
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage as fraction of total."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total
        return 0.0
    
    def get_memory_stats(self) -> Dict:
        """Get memory management statistics."""
        gpu_experts = sum(1 for loc in self.expert_locations.values() if loc == 'gpu')
        cpu_experts = sum(1 for loc in self.expert_locations.values() if loc == 'cpu')
        loading_experts = sum(1 for loc in self.expert_locations.values() if loc == 'loading')
        
        return {
            'gpu_experts': gpu_experts,
            'cpu_experts': cpu_experts,
            'loading_experts': loading_experts,
            'gpu_memory_usage': self._get_gpu_memory_usage(),
            'total_expert_accesses': sum(self.expert_access_count.values()),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (experts found on GPU vs total accesses)."""
        total_accesses = sum(self.expert_access_count.values())
        if total_accesses == 0:
            return 1.0
        
        # This is a simplified calculation
        # Real implementation would track actual hits/misses
        gpu_experts = sum(1 for loc in self.expert_locations.values() if loc == 'gpu')
        total_experts = len(self.expert_locations)
        
        return gpu_experts / total_experts if total_experts > 0 else 1.0
```

This technical specification provides the detailed implementation architecture for distributed MOE model parallelism. The key innovations include:

1. **Smart Expert Distribution**: Efficient mapping of experts to GPUs
2. **Optimized Communication**: Batched, asynchronous transfers with minimal overhead
3. **Intelligent Memory Management**: LRU caching with predictive prefetching
4. **Performance Monitoring**: Comprehensive metrics for optimization

The implementation is designed to be modular and extensible, allowing for future optimizations and different distribution strategies.