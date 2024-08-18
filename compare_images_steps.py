import os
import cv2
import re
import numpy as np
from sewar.full_ref import mse, rmse, psnr, ssim, uqi, msssim, ergas, scc, rase, sam, vifp
import json
import matplotlib.pyplot as plt

chart_data = {}


# Define a function to load images using OpenCV
def load_image(image_path):
    return cv2.imread(image_path)

# Convert BGR to RGB
def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Calculate all metrics
def calculate_metrics(image1, image2):

    # SSIM
    metrics = ssim(image1, image2)[0]  # sewar.ssim returns a tuple (score, _)

    
    return metrics

# Function to extract sampler, scheduler, and step from filename
def extract_info(filename):
    match = re.match(r'(.+?)_(.+?)_(\d+)_.*\.png', filename)
    if match:
        return match.groups()  # (sampler, scheduler, step)
    return None

# Group images by sampler and scheduler
def group_images_by_sampler_scheduler(image_dir):
    grouped_images = {}
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            info = extract_info(filename)
            if info:
                sampler, scheduler, step = info
                key = f"{sampler}_{scheduler}"
                if key not in grouped_images:
                    grouped_images[key] = []
                grouped_images[key].append((int(step), os.path.join(image_dir, filename)))
    return grouped_images

# Compare images in step ranges
def compare_image_pairs(image_pairs):
    results = {}
    for step_range, (img_path1, img_path2) in image_pairs.items():
        # Load images
        image1 = load_image(img_path1)
        image2 = load_image(img_path2)
        
        # Convert to RGB
        image1_rgb = convert_to_rgb(image1)
        image2_rgb = convert_to_rgb(image2)
        
        # Calculate metrics
        metrics = calculate_metrics(image1_rgb, image2_rgb)
        results[step_range] = metrics
        
        print(f'Comparison for steps {step_range}:')
    for metric, value in results.items():
        print(f'{metric}: {value}')
    print('\n')
    
    return results

def plot_metrics(chart_data):
   
    # Extract step ranges and MSE values
    plt.figure(figsize=(10, 6))  
    for key, results in chart_data.items():
        # Extract step ranges and MSE values
        step_ranges = list(results.keys())
        mse_values = list(results.values())

        plt.plot(step_ranges, mse_values, linestyle='-', alpha=0.7, label=key)

    plt.xticks(range(min(step_ranges), max(step_ranges) + 1, 5))
    plt.xlabel('Sampling Step')
    plt.ylabel("SSIM Value")
    plt.title(f'Metric for each sampler_scheduler')
    plt.legend()
    #plt.grid(True)
    plt.savefig(f'converge.png')
    #plt.show()

# Main function to compare images
def main(image_dir):
    grouped_images = group_images_by_sampler_scheduler(image_dir)
    
    for key, images in grouped_images.items():
        # Sort images by step
        images.sort()
        # Compare images in step ranges
        image_pairs = {}
        for i in range(0, len(images) -1):
            step_range = images[i + 1][0]
            image_pairs[step_range] = (images[i][1], images[i + 1][1])
        
        print(f'Comparing images for {key}:')
        results = compare_image_pairs(image_pairs)
        chart_data.update({key: results})
        #print(new_data)
        # Optionally, save results to a file
        #with open(f'comparison_results_{key}.json', 'w') as f:
        #    json.dump(results, f, indent=4)
    
    plot_metrics(chart_data)

# Directory containing the images
image_dir = '/home/sivar/ComfyUI-master/output/converge'

# Run the comparison
main(image_dir)
