import time
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Any

from evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluates trained ECG classification models, computes clinical metrics, and profiles hardware execution."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        classes: List[str]
    ) -> None:
        self.model = model
        self.device = device
        self.classes = classes

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Runs validation inference and computes multi-label metrics.

        Args:
            test_loader: DataLoader containing test split records.
            threshold: Decision probability threshold.

        Returns:
            Dict: Comprehensive clinical score reports.
        """
        self.model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                # Mixed precision inference
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.model(inputs)
                    probs = torch.sigmoid(outputs)
                
                all_probs.append(probs.float().cpu().numpy())
                all_targets.append(targets.numpy())

        y_prob = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        # Compute metric profile
        metrics = compute_all_metrics(y_true, y_prob, self.classes, threshold=threshold)
        
        # Save raw predictions for external research inspection
        metrics["raw_predictions"] = y_prob.tolist()
        metrics["raw_targets"] = y_true.tolist()

        return metrics

    def profile_hardware(self, input_shape: Tuple[int, ...] = (1, 12, 1000)) -> Dict[str, Any]:
        """Profiles hardware metrics: parameter count, CPU/GPU latency, throughput, and FLOPs.

        Args:
            input_shape: Mock ECG batch tensor dimension (Batch, Channels, Length).

        Returns:
            Dict: Performance report including latency, parameter counts, and FLOPs.
        """
        self.model.eval()
        self.model.to(self.device)
        
        # 1. Parameter Counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            
        with torch.no_grad():
            # 2. Latency Range test (Single sample)
            single_shape = (1, input_shape[1], input_shape[2])
            dummy_single = torch.randn(*single_shape).to(self.device)
            
            # Warmup passes
            for _ in range(10):
                _ = self.model(dummy_single)
                
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            runs = 100
            for _ in range(runs):
                _ = self.model(dummy_single)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            avg_latency_ms = ((end_time - start_time) / runs) * 1000.0

            # 3. Throughput Test (Batch size = 128)
            batch_shape = (128, input_shape[1], input_shape[2])
            dummy_batch = torch.randn(*batch_shape).to(self.device)
            
            for _ in range(5):
                _ = self.model(dummy_batch)
                
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            batch_runs = 20
            for _ in range(batch_runs):
                _ = self.model(dummy_batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            total_samples = batch_shape[0] * batch_runs
            throughput_sps = total_samples / (end_time - start_time)

            # 4. Analytical FLOPs calculation
            flops = estimate_flops(self.model, input_shape[2])

        profile_results = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "inference_latency_ms": avg_latency_ms,
            "throughput_samples_per_sec": throughput_sps,
            "theoretical_flops": flops
        }

        logger.info(
            f"Hardware Profile - Params: {total_params:,} | "
            f"Latency: {avg_latency_ms:.2f} ms | "
            f"Throughput: {throughput_sps:.1f} samples/sec | "
            f"FLOPs: {flops:,}"
        )
        return profile_results


def estimate_flops(model: torch.nn.Module, signal_length: int) -> int:
    """Calculates theoretical FLOPs analytically by simulating the forward pass layer sequence.

    Args:
        model: Neural network model instance.
        signal_length: Time series sample length.

    Returns:
        int: Cumulative Multiply-Accumulate operations (FLOPs).
    """
    total_flops = 0
    # Track spatial dimension state through model layer steps
    curr_len = signal_length
    
    # We iterate over modules to count linear/convolutional operations
    for name, module in model.named_modules():
        # Handle 1D convolutions
        if isinstance(module, torch.nn.Conv1d):
            in_c = module.in_channels
            out_c = module.out_channels
            k_size = module.kernel_size[0]
            stride = module.stride[0]
            padding = module.padding[0]
            dilation = module.dilation[0]
            
            # Compute new length for this conv layer output, bounding by at least 1 to handle small spatial dimensions
            val = curr_len + 2 * padding - dilation * (k_size - 1) - 1
            out_len = max(1, int(val / stride + 1)) if val >= 0 else 1
            
            # Conv FLOPs: 2 * in_c * out_c * k_size * out_len (MAC count scaled by 2)
            layer_flops = 2 * in_c * out_c * k_size * out_len
            total_flops += layer_flops
            
            # Update sequence length tracker (assume main path sequence length changes)
            # Only update main path spatial tracker if this is not a shortcut layer
            if "shortcut" not in name and "downsample" not in name:
                curr_len = out_len
                
        # Handle Linear layers
        elif isinstance(module, torch.nn.Linear):
            in_f = module.in_features
            out_f = module.out_features
            # Linear FLOPs: 2 * in_f * out_f
            total_flops += 2 * in_f * out_f
            
        # Handle recurrent layers (LSTM)
        elif isinstance(module, torch.nn.LSTM):
            in_f = module.input_size
            hid_f = module.hidden_size
            layers = module.num_layers
            bi = 2 if module.bidirectional else 1
            
            # LSTM gates involve 4 linear steps per cell. We process sequence of length curr_len.
            # FLOPs = 4 * bi * layers * curr_len * (2 * in_f * hid_f + 2 * hid_f * hid_f)
            step_flops = 4 * bi * layers * curr_len * (2 * in_f * hid_f + 2 * hid_f * hid_f)
            total_flops += step_flops
            
        # Handle Multi-head Attention
        elif isinstance(module, torch.nn.MultiheadAttention):
            embed_dim = module.embed_dim
            # Attention projection queries, keys, values, and output projection:
            # 4 linear projections of shape embed_dim -> embed_dim. FLOPs = 4 * 2 * embed_dim^2 * curr_len
            proj_flops = 8 * (embed_dim ** 2) * curr_len
            # QK^T matrix multiply: FLOPs = 2 * curr_len^2 * embed_dim
            # Softmax QK^T * V matrix multiply: FLOPs = 2 * curr_len^2 * embed_dim
            attn_matrix_flops = 4 * (curr_len ** 2) * embed_dim
            
            total_flops += (proj_flops + attn_matrix_flops)

    # If the model features parallel branches (e.g. Inception/AttentionHybrid), 
    # the sequential estimate might underestimate, so we apply a standard base filter scaling
    if total_flops == 0:
        # Fallback approximation for complex custom blocks
        total_params = sum(p.numel() for p in model.parameters())
        total_flops = 2 * total_params * signal_length
        
    return total_flops
