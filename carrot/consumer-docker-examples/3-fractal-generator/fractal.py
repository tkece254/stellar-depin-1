#!/usr/bin/env python3
"""
Mandelbrot Fractal Generator
Generates high-resolution fractal image with custom coloring
Execution time: 10-20 seconds depending on resolution
"""
import numpy as np
from PIL import Image
import time
import sys

def mandelbrot(c, max_iter):
    """Calculate Mandelbrot set value for complex number c"""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def generate_fractal(width=2000, height=2000, max_iter=256):
    """Generate Mandelbrot fractal image"""
    print(f"Generating {width}x{height} fractal with {max_iter} iterations...")
    start_time = time.time()

    # Define the complex plane boundaries
    xmin, xmax = -2.5, 1.0
    ymin, ymax = -1.25, 1.25

    # Create coordinate arrays
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    # Initialize image array
    fractal = np.zeros((height, width))

    # Calculate fractal (this is the compute-intensive part)
    total_pixels = width * height
    progress_step = total_pixels // 10

    print("Computing fractal values...")
    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            fractal[i, j] = mandelbrot(c, max_iter)

            # Show progress
            pixel_num = i * width + j
            if pixel_num % progress_step == 0:
                progress = (pixel_num / total_pixels) * 100
                elapsed = time.time() - start_time
                print(f"Progress: {progress:.1f}% ({elapsed:.1f}s elapsed)")

    print("Applying color mapping...")
    # Normalize and apply color map
    fractal_normalized = (fractal / max_iter * 255).astype(np.uint8)

    # Create RGB image with custom coloring
    img = Image.fromarray(fractal_normalized, mode='L')
    img = img.convert('RGB')

    # Apply color gradient (blue to yellow to red)
    pixels = img.load()
    for i in range(height):
        for j in range(width):
            gray_val = fractal_normalized[i, j]
            if gray_val < 85:
                # Blue to cyan
                r = 0
                g = int(gray_val * 3)
                b = 255
            elif gray_val < 170:
                # Cyan to yellow
                gray_val -= 85
                r = int(gray_val * 3)
                g = 255
                b = int(255 - gray_val * 3)
            else:
                # Yellow to red
                gray_val -= 170
                r = 255
                g = int(255 - gray_val * 3)
                b = 0
            pixels[j, i] = (r, g, b)

    computation_time = time.time() - start_time
    print(f"Fractal generated in {computation_time:.2f} seconds")

    return img, computation_time

def main():
    print("=" * 60)
    print("MANDELBROT FRACTAL GENERATOR")
    print("=" * 60)

    # High resolution for 10-20 second execution
    width = 2000
    height = 2000
    max_iterations = 256

    # Generate fractal
    fractal_image, exec_time = generate_fractal(width, height, max_iterations)

    # Save output
    output_file = "/output/fractal.png"
    print(f"\nSaving fractal to {output_file}...")
    fractal_image.save(output_file)

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Resolution: {width}x{height} pixels")
    print(f"Iterations: {max_iterations}")
    print(f"Execution time: {exec_time:.2f} seconds")
    print(f"Output file: {output_file}")
    print(f"File size: {fractal_image.size}")
    print("=" * 60)

    # Return result hash for blockchain
    result_hash = f"fractal_{width}x{height}_{max_iterations}iter_{exec_time:.2f}s"
    print(f"\nRESULT:{result_hash}")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
