# Calculate PSNR between baseline and all ablation study experiments

import argparse
import os
import logging
import json
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import torchvision.transforms.functional as tf

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def readImages(image_dir):
    """Load all PNG images from a directory"""
    images = {}
    image_dir = Path(image_dir)
    if not image_dir.exists():
        logging.warning(f"Directory {image_dir} does not exist")
        return images

    png_files = list(image_dir.glob("*.png"))
    if not png_files:
        logging.warning(f"No PNG files found in {image_dir}")
        return images

    for img_path in tqdm(png_files, desc=f"Loading images from {image_dir.name}"):
        try:
            image = Image.open(img_path)
            images[img_path.name] = tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].cuda()
        except Exception as e:
            logging.warning(f"Failed to load {img_path.name}: {e}")
    return images


def calculate_psnr_between_dirs(baseline_images, test_images, test_name):
    """Calculate PSNR between baseline and test images"""
    common_images = set(baseline_images.keys()).intersection(set(test_images.keys()))

    if not common_images:
        logging.warning(f"No common images found between baseline and {test_name}")
        return None, 0

    psnr_values = []
    for image_name in common_images:
        img1 = baseline_images[image_name]
        img2 = test_images[image_name]

        if img1.shape != img2.shape:
            logging.warning(
                f"Image {image_name} has different dimensions in {test_name}, skipping."
            )
            continue

        psnr_value = psnr(img1, img2).item()
        psnr_values.append(psnr_value)

    if psnr_values:
        average_psnr = sum(psnr_values) / len(psnr_values)
        return average_psnr, len(psnr_values)
    else:
        return None, 0


def find_experiment_directories(results_dir):
    """Find all experiment directories (excluding baseline and configs)"""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")

    experiment_dirs = []
    for item in results_path.iterdir():
        if item.is_dir() and item.name not in ["baseline", "configs"]:
            experiment_dirs.append(item)

    return sorted(experiment_dirs)


def run_ablation_psnr_analysis(results_dir):
    """Run PSNR analysis for all ablation study experiments"""
    results_path = Path(results_dir)
    baseline_dir = results_path / "baseline"

    if not baseline_dir.exists():
        raise ValueError(f"Baseline directory {baseline_dir} does not exist")

    # Load baseline images once
    logging.info("Loading baseline images...")
    baseline_images = readImages(baseline_dir)

    if not baseline_images:
        raise ValueError("No images found in baseline directory")

    # Find all experiment directories
    experiment_dirs = find_experiment_directories(results_dir)

    if not experiment_dirs:
        logging.warning("No experiment directories found")
        return

    # Calculate PSNR for each experiment
    results = []
    logging.info(f"Found {len(experiment_dirs)} experiments to compare")

    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        logging.info(f"Calculating PSNR for {exp_name}...")

        # Load experiment images
        exp_images = readImages(exp_dir)

        if not exp_images:
            logging.warning(f"No images found in {exp_name}, skipping")
            continue

        # Calculate PSNR
        avg_psnr, num_images = calculate_psnr_between_dirs(
            baseline_images, exp_images, exp_name
        )

        if avg_psnr is not None:
            results.append(
                {"experiment": exp_name, "psnr": avg_psnr, "num_images": num_images}
            )
            logging.info(f"{exp_name}: {avg_psnr:.2f} dB ({num_images} images)")
        else:
            logging.warning(f"Failed to calculate PSNR for {exp_name}")

    # Print summary
    print("\n" + "=" * 60)
    print("PSNR ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Baseline: {baseline_dir}")
    print(f"Total experiments: {len(results)}")
    print()

    if results:
        # Sort by PSNR (descending)
        results.sort(key=lambda x: x["psnr"], reverse=True)

        print("Results (sorted by PSNR):")
        print("-" * 40)
        for result in results:
            print(
                f"{result['experiment']:25s}: {result['psnr']:6.2f} dB ({result['num_images']} images)"
            )

        # Save results to JSON
        results_file = results_path / "psnr_analysis.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "baseline_dir": str(baseline_dir),
                    "total_experiments": len(results),
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {results_file}")
    else:
        print("No valid results found")


def run_single_comparison(dir1, dir2):
    """Run single directory comparison (original functionality)"""
    dir1 = Path(dir1)
    dir2 = Path(dir2)

    if not dir1.is_dir() or not dir2.is_dir():
        raise ValueError("Both arguments must be valid directories.")

    logging.info(f"Loading images from {dir1}...")
    images1 = readImages(dir1)
    logging.info(f"Loading images from {dir2}...")
    images2 = readImages(dir2)

    avg_psnr, num_images = calculate_psnr_between_dirs(images1, images2, dir2.name)

    if avg_psnr is not None:
        print(f"Average PSNR: {avg_psnr:.2f} dB ({num_images} images)")
    else:
        logging.error("No valid images found for comparison.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate PSNR between baseline and ablation study experiments."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to results directory (e.g., results/jelly) containing baseline and experiment subdirectories.",
    )
    parser.add_argument(
        "--compare-dirs",
        nargs=2,
        metavar=("DIR1", "DIR2"),
        help="Compare two specific directories instead of running ablation analysis.",
    )

    args = parser.parse_args()

    if args.compare_dirs:
        # Original functionality: compare two specific directories
        run_single_comparison(args.compare_dirs[0], args.compare_dirs[1])
    else:
        # New functionality: analyze all experiments in results directory
        run_ablation_psnr_analysis(args.results_dir)
