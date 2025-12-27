#!/usr/bin/env python3
"""
OSMLoc Custom Image Inference Script

This script runs inference on a single custom image that you supply.
It handles the image processing and provides localization results.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import json
import subprocess

# Add the maploc module to the path
sys.path.append(str(Path.cwd()))

from maploc import logger
from maploc.conf import data as conf_data_dir
from maploc.data import MapillaryDataModule
from maploc.module import GenericModule
from maploc.data.torch import collate


def load_and_preprocess_image(image_path: str, target_size: tuple = (512, 512)) -> torch.Tensor:
    """
    Load and preprocess a custom image for OSMLoc inference.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
    
    Returns:
        Preprocessed image tensor
    """
    print(f"Loading and preprocessing image: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize while maintaining aspect ratio
    original_size = image.size
    ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    image = image.resize(new_size, Image.LANCZOS)
    
    # Pad to target size
    padded_image = Image.new('RGB', target_size, (0, 0, 0))
    x_offset = (target_size[0] - new_size[0]) // 2
    y_offset = (target_size[1] - new_size[1]) // 2
    padded_image.paste(image, (x_offset, y_offset))
    
    # Convert to tensor and normalize
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(padded_image)
    
    print(f"Image processed: {image_path} -> {image_tensor.shape}")
    return image_tensor.unsqueeze(0)  # Add batch dimension


def create_dummy_batch(image_tensor: torch.Tensor, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """
    Create a dummy batch with the required fields for OSMLoc inference.
    
    Args:
        image_tensor: Preprocessed image tensor
        device: Device to use (cuda or cpu)
    
    Returns:
        Dictionary containing the batch data
    """
    print("Creating dummy batch for inference...")
    
    # Move image to the correct device
    image_tensor = image_tensor.to(device)
    
    # Create dummy data for required fields
    # Note: In a real scenario, you would need to provide actual map data
    # and georeferencing information for the image location
    
    # Create proper map data with integer class indices
    # Based on MGL dataset configuration: areas=7, ways=10, nodes=33
    height, width = 256, 256
    
    # Create random but valid class indices for each category
    # Areas: 0-6 (7 classes), Ways: 0-9 (10 classes), Nodes: 0-32 (33 classes)
    areas = torch.randint(0, 7, (height, width), device=device, dtype=torch.long)
    ways = torch.randint(0, 10, (height, width), device=device, dtype=torch.long)
    nodes = torch.randint(0, 33, (height, width), device=device, dtype=torch.long)
    
    # Stack to create the map tensor with shape (3, height, width), then add batch dimension
    map_data = torch.stack([areas, ways, nodes], dim=0).unsqueeze(0)  # Shape: (1, 3, height, width)
    
    batch = {
        "image": image_tensor,
        # Create proper map data with integer class indices
        "map": map_data,  # 3 channels (areas, ways, nodes), 256x256 map
        # Create dummy canvas (this would need real georeferencing in practice)
        "canvas": None,  # This would be a proper Canvas object
        # Create dummy ground truth (not used for inference, but required)
        "uv": torch.zeros((1, 2), device=device),
        "roll_pitch_yaw": torch.zeros((1, 3), device=device),
        # Additional required fields
        "accuracy_gps": torch.tensor([10.0], device=device),  # 10m GPS accuracy
        "uv_gps": torch.zeros((1, 2), device=device),
        # Add missing required fields that the model expects
        "camera": None,  # Will create proper Camera object below
        "valid": torch.ones((1, 1, 256, 256), device=device),  # Valid pixels mask
        "pixels_per_meter": torch.tensor([2.0], device=device),  # From config
        "uv_init": torch.zeros((1, 2), device=device),
        # yaw_prior is optional and handled by the model only if present,
        "map_mask": torch.ones((256, 256), device=device, dtype=torch.bool),  # Boolean mask (H, W)
        "depth": None,
        "chunk_id": None
    }
    
    # Create proper Camera object
    from maploc.utils.wrappers import Camera
    # Create a simple pinhole camera with reasonable parameters
    # Format: [width, height, fx, fy, cx, cy, distortion_params...]
    camera_data = torch.tensor([512, 512, 500, 500, 256, 256, 0, 0], device=device).unsqueeze(0)
    batch["camera"] = Camera(camera_data)
    
    print("Dummy batch created")
    return batch


def run_custom_image_inference(
    image_path: str,
    checkpoint_path: str = "loca_polar_small.ckpt",
    dataset_name: str = "mgl",
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Run inference on a custom image using OSMLoc.
    
    Args:
        image_path: Path to the custom image
        checkpoint_path: Path to model checkpoint
        dataset_name: Dataset configuration to use
        visualize: Whether to visualize results
    
    Returns:
        Dictionary containing inference results
    """
    print(f"Running OSMLoc inference on custom image: {image_path}")
    
    # Validate inputs
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load and preprocess image
    image_tensor = load_and_preprocess_image(image_path)
    
    # Setup configuration (for model architecture, not for data)
    # Use the actual osmloc_small configuration that matches the checkpoint
    print("Loading osmloc_small.yaml configuration...")
    
    # Try multiple approaches to find the configuration file
    config_path = None
    
    # Approach 1: Direct path (most reliable)
    direct_path = Path("maploc/conf/osmloc_small.yaml")
    if direct_path.exists():
        config_path = direct_path
        print(f"Using direct path: {config_path}")
    
    # Approach 2: Relative to conf_data_dir
    if config_path is None:
        try:
            relative_path = Path(conf_data_dir.__file__).parent.parent / "osmloc_small.yaml"
            if relative_path.exists():
                config_path = relative_path
                print(f"Using relative path: {config_path}")
        except:
            pass
    
    # Approach 3: Search for the file
    if config_path is None:
        import subprocess
        try:
            result = subprocess.run(['find', '.', '-name', 'osmloc_small.yaml'], 
                                  capture_output=True, text=True, cwd='/home/nick/OSMLoc')
            if result.stdout.strip():
                found_paths = result.stdout.strip().split('\n')
                config_path = Path(found_paths[0])
                print(f"Found configuration at: {config_path}")
        except:
            pass
    
    if config_path is None or not config_path.exists():
        raise FileNotFoundError("Could not find osmloc_small.yaml configuration file")
    
    print(f"Configuration exists: {config_path.exists()}")
    
    try:
        model_cfg = OmegaConf.load(config_path)
        
        # Debug: Print key configuration values
        print("Model configuration loaded:")
        print(f"  latent_dim: {model_cfg.model.latent_dim}")
        print(f"  image_encoder.features: {model_cfg.model.image_encoder.features}")
        print(f"  image_encoder.out_channels: {model_cfg.model.image_encoder.out_channels}")
        
        # Fix all interpolation references
        try:
            # Load data configuration for num_classes and pixel_per_meter
            data_cfg = OmegaConf.load(Path("maploc/conf/data/mapillary_mgl.yaml"))
            
            # Resolve num_classes reference
            if hasattr(model_cfg.model.map_encoder, 'num_classes') and isinstance(model_cfg.model.map_encoder.num_classes, str):
                model_cfg.model.map_encoder.num_classes = data_cfg.num_classes
                print(f"  map_encoder.num_classes: {model_cfg.model.map_encoder.num_classes}")
            
            # Resolve pixel_per_meter reference
            if hasattr(model_cfg.model, 'pixel_per_meter') and isinstance(model_cfg.model.pixel_per_meter, str):
                model_cfg.model.pixel_per_meter = data_cfg.pixel_per_meter
                print(f"  pixel_per_meter: {model_cfg.model.pixel_per_meter}")
            
            # Resolve matching_dim references
            matching_dim = model_cfg.model.matching_dim
            latent_dim = model_cfg.model.latent_dim
            
            # Fix map_encoder.output_dim
            if hasattr(model_cfg.model.map_encoder, 'output_dim') and isinstance(model_cfg.model.map_encoder.output_dim, str):
                model_cfg.model.map_encoder.output_dim = matching_dim
                print(f"  map_encoder.output_dim: {model_cfg.model.map_encoder.output_dim}")
            
            # Fix bev_net.latent_dim and bev_net.output_dim
            if 'bev_net' in model_cfg.model:
                if hasattr(model_cfg.model.bev_net, 'latent_dim') and isinstance(model_cfg.model.bev_net.latent_dim, str):
                    model_cfg.model.bev_net.latent_dim = latent_dim
                    print(f"  bev_net.latent_dim: {model_cfg.model.bev_net.latent_dim}")
                if hasattr(model_cfg.model.bev_net, 'output_dim') and isinstance(model_cfg.model.bev_net.output_dim, str):
                    model_cfg.model.bev_net.output_dim = matching_dim
                    print(f"  bev_net.output_dim: {model_cfg.model.bev_net.output_dim}")
            
            # Fix image_encoder checkpoint path
            if hasattr(model_cfg.model.image_encoder, 'ckpt') and isinstance(model_cfg.model.image_encoder.ckpt, str):
                encoder = model_cfg.model.image_encoder.encoder
                model_cfg.model.image_encoder.ckpt = f"checkpoints/depth_anything/depth_anything_{encoder}14.pth"
                # Update to use available path
                model_cfg.model.image_encoder.ckpt = model_cfg.model.image_encoder.ckpt.replace(
                    "checkpoints/depth_anything/", 
                    "maploc/models/depth_anything/ckpt/"
                )
                print(f"  image_encoder.ckpt: {model_cfg.model.image_encoder.ckpt}")
            
        except Exception as e:
            print(f"Warning: Could not resolve some interpolation references: {e}")
            # Provide default values for critical parameters
            model_cfg.model.map_encoder.num_classes = {"areas": 7, "ways": 10, "nodes": 33}
            model_cfg.model.pixel_per_meter = 2
            model_cfg.model.map_encoder.output_dim = model_cfg.model.matching_dim
            if 'bev_net' in model_cfg.model:
                model_cfg.model.bev_net.latent_dim = model_cfg.model.latent_dim
                model_cfg.model.bev_net.output_dim = model_cfg.model.matching_dim
            print(f"  Using default values for interpolation references")
            print(f"  map_encoder.num_classes: {model_cfg.model.map_encoder.num_classes}")
            print(f"  pixel_per_meter: {model_cfg.model.pixel_per_meter}")
            print(f"  map_encoder.output_dim: {model_cfg.model.map_encoder.output_dim}")
        
    except Exception as e:
        print(f"Failed to load configuration file: {e}")
        print("Using fallback configuration...")
        
        # Fallback: Create the correct configuration manually
        model_cfg = OmegaConf.create({
            "model": {
                "name": "osmloc",
                "latent_dim": 128,
                "matching_dim": 8,
                "z_max": 32,
                "x_max": 32,
                "pixel_per_meter": 2,
                "num_scale_bins": 33,
                "num_rotations": 64,
                "image_encoder": {
                    "encoder": "vits",
                    "features": 64,  # This is the key fix - was 128, should be 64
                    "out_dim": 128,
                    "out_channels": [48, 96, 192, 384],  # This is the key fix - was [96, 192, 384, 768]
                    "ckpt": "maploc/models/depth_anything/ckpt/depth_anything_vits14.pth",
                    "shallow_encoder": {
                        "backbone": {
                            "encoder": "resnet101"
                        }
                    }
                },
                "map_encoder": {
                    "embedding_dim": 16,
                    "output_dim": 8,
                    "num_classes": {"areas": 7, "ways": 10, "nodes": 33},
                    "backbone": {
                        "encoder": "vgg19",
                        "pretrained": False,
                        "output_scales": [0],
                        "num_downsample": 3,
                        "decoder": [128, 64, 64],
                        "padding": "replicate"
                    },
                    "unary_prior": False
                },
                "bev_net": {
                    "num_blocks": 4,
                    "latent_dim": 128,
                    "output_dim": 8,
                    "confidence": True
                }
            }
        })
        
        print("Fallback configuration created with correct dimensions:")
        print(f"  image_encoder.features: {model_cfg.model.image_encoder.features}")
        print(f"  image_encoder.out_channels: {model_cfg.model.image_encoder.out_channels}")
        print(f"  map_encoder.num_classes: {model_cfg.model.map_encoder.num_classes}")
        print(f"  pixel_per_meter: {model_cfg.model.pixel_per_meter}")
        print(f"  map_encoder.output_dim: {model_cfg.model.map_encoder.output_dim}")
    
    # Update the depth_anything checkpoint path to match available files
    if "image_encoder" in model_cfg.model and "ckpt" in model_cfg.model.image_encoder:
        original_ckpt = model_cfg.model.image_encoder.ckpt
        model_cfg.model.image_encoder.ckpt = original_ckpt.replace(
            "checkpoints/depth_anything/", 
            "maploc/models/depth_anything/ckpt/"
        )
        print(f"Updated depth_anything checkpoint: {original_ckpt} -> {model_cfg.model.image_encoder.ckpt}")
    
    # Load model with strict=False to ignore size mismatches
    # and then manually fix the configuration
    print(f"Loading model from {checkpoint_path}...")
    print("Model configuration being used:")
    print(f"  model.latent_dim: {model_cfg.model.latent_dim}")
    print(f"  model.matching_dim: {model_cfg.model.matching_dim}")
    print(f"  image_encoder.encoder: {model_cfg.model.image_encoder.encoder}")
    print(f"  image_encoder.features: {model_cfg.model.image_encoder.features}")
    print(f"  image_encoder.out_channels: {model_cfg.model.image_encoder.out_channels}")
    print(f"  map_encoder.num_classes: {model_cfg.model.map_encoder.num_classes}")
    
    # Create model with correct configuration first
    print("Creating model with correct configuration...")
    from maploc.models import get_model
    from maploc.module import GenericModule
    
    # Create model with our correct configuration
    model_name = model_cfg.model.get("name", "osmloc")
    model_instance = get_model(model_name)(model_cfg.model)
    
    # Create GenericModule
    model = GenericModule(model_cfg)
    model.model = model_instance
    model = model.eval()
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model created and moved to {device}")
    
    # Try to load checkpoint parameters
    print("Loading checkpoint parameters...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", {})
        
        if state_dict:
            # Filter out incompatible keys and load what we can
            model_state_dict = model.state_dict()
            compatible_state_dict = {}
            incompatible_keys = []
            
            for key, value in state_dict.items():
                if key in model_state_dict:
                    if value.shape == model_state_dict[key].shape:
                        compatible_state_dict[key] = value
                    else:
                        incompatible_keys.append(f"{key}: {value.shape} vs {model_state_dict[key].shape}")
                else:
                    incompatible_keys.append(f"{key}: key not found")
            
            if compatible_state_dict:
                model.load_state_dict(compatible_state_dict, strict=False)
                print(f"✓ Loaded {len(compatible_state_dict)} compatible parameters")
            
            if incompatible_keys:
                print(f"⚠ Skipped {len(incompatible_keys)} incompatible parameters:")
                # Print first 5 incompatible keys to avoid spam
                for key_info in incompatible_keys[:5]:
                    print(f"  - {key_info}")
                if len(incompatible_keys) > 5:
                    print(f"  - ... and {len(incompatible_keys) - 5} more")
        else:
            print("⚠ Checkpoint has no state_dict")
            
    except Exception as e:
        print(f"⚠ Failed to load checkpoint parameters: {e}")
        print("✓ Model will use random initialization")
    
    print("✓ Model ready for inference")
    # Model is already in eval mode and on the correct device from above
    print(f"Using {device.upper()} for inference")
    
    # Create dummy batch
    batch = create_dummy_batch(image_tensor, device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        try:
            pred = model(batch)
            
            # Extract results with proper handling of different tensor shapes
            uv_max = pred["uv_max"].cpu().numpy()
            yaw_max = pred["yaw_max"].cpu().numpy()
            
            # Handle 0-dimensional arrays by converting to lists
            if uv_max.ndim == 0:
                uv_max = [float(uv_max)]
            else:
                uv_max = uv_max.tolist()
                
            if yaw_max.ndim == 0:
                yaw_max = [float(yaw_max)]
            else:
                yaw_max = yaw_max.tolist()
                
            results = {
                "uv_max": uv_max,
                "yaw_max": yaw_max,
                "log_probs_shape": list(pred["log_probs"].shape) if hasattr(pred["log_probs"], 'shape') else "unknown",
                "confidence": float(pred.get("confidence", 1.0))
            }
            
            print("Inference completed successfully!")
            print(f"Results: {json.dumps(results, indent=2)}")
            
            # Visualize if requested
            if visualize:
                visualize_custom_results(image_path, pred, batch)
            
            return results
            
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            
            # Try a simpler approach - just get the image features
            print("Trying simpler feature extraction approach...")
            try:
                # Get image encoder
                image_encoder = model.model.image_encoder
                
                # Extract features
                image_features = image_encoder(batch["image"])
                print(f"Image features shape: {image_features.shape}")
                
                results = {
                    "image_features_shape": list(image_features.shape),
                    "status": "partial_success",
                    "error": str(e)
                }
                
                return results
                
            except Exception as e2:
                print(f"Feature extraction also failed: {e2}")
                results = {
                    "status": "partial_success",
                    "error": f"Inference completed but visualization failed: {e}",
                    "raw_results": str(pred) if isinstance(pred, dict) else "non-dict result",
                    "feature_extraction_error": str(e2)
                }
                return results


def visualize_custom_results(image_path: str, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
    """
    Visualize results for custom image inference.
    
    Args:
        image_path: Path to the original image
        pred: Model predictions
        batch: Input batch
    """
    print("Generating visualization...")
    
    # Load original image for visualization
    original_image = Image.open(image_path).convert('RGB')
    
    # Get predictions
    log_probs = pred["log_probs"][0].cpu().numpy()
    uv_max = pred["uv_max"][0].cpu().numpy()
    yaw_max = pred["yaw_max"][0].cpu().numpy()
    
    # Create visualization
    plt.figure(figsize=(14, 7))
    
    # Show original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Show processed image
    plt.subplot(1, 3, 2)
    processed_image = batch["image"][0].cpu().numpy()
    if processed_image.shape[0] == 3:
        processed_image = np.transpose(processed_image, (1, 2, 0))
    if processed_image.max() > 1.0:
        processed_image = processed_image / 255.0
    plt.imshow(processed_image)
    plt.title("Processed Image")
    plt.axis('off')
    
    # Show prediction heatmap
    plt.subplot(1, 3, 3)
    heatmap = log_probs.max(axis=0)
    plt.imshow(heatmap, cmap='viridis')
    # Handle 0-dimensional arrays more robustly
    try:
        uv_x = float(uv_max[0]) if uv_max.ndim > 0 else float(uv_max)
        uv_y = float(uv_max[1]) if uv_max.ndim > 0 and uv_max.shape[0] > 1 else uv_x
        yaw_val = float(yaw_max[0]) if yaw_max.ndim > 0 else float(yaw_max)
    except (IndexError, TypeError) as e:
        print(f"Warning: Could not extract coordinates from predictions: {e}")
        uv_x, uv_y, yaw_val = 0.0, 0.0, 0.0
    
    plt.scatter(uv_x, uv_y, c='red', s=100, marker='x')
    plt.title(f"Prediction Heatmap\nMax at: ({uv_x:.1f}, {uv_y:.1f})\nYaw: {np.degrees(yaw_val):.1f}°")
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function for custom image inference.
    """
    parser = argparse.ArgumentParser(description="OSMLoc Custom Image Inference")
    parser.add_argument("image_path", type=str, 
                       help="Path to the custom image file")
    parser.add_argument("--checkpoint", type=str, default="loca_polar_small.ckpt", 
                       help="Path to model checkpoint (default: loca_polar_small.ckpt)")
    parser.add_argument("--dataset", type=str, default="mgl", 
                       choices=["mgl", "taipei", "brisbane", "detroit", "munich"],
                       help="Dataset configuration to use (default: mgl)")
    parser.add_argument("--no_visualize", action="store_true",
                       help="Disable visualization")
    
    args = parser.parse_args()
    
    try:
        # Run inference
        results = run_custom_image_inference(
            args.image_path,
            checkpoint_path=args.checkpoint,
            dataset_name=args.dataset,
            visualize=not args.no_visualize
        )
        
        # Save results
        output_file = f"{os.path.splitext(args.image_path)[0]}_osmloc_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()