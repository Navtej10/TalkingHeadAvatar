import time
import cv2
import numpy as np

def benchmark_poisson(frame, refined_crop, by1, by2, bx1, bx2):
    start = time.perf_counter()
    center = (bx1 + (bx2 - bx1) // 2, by1 + (by2 - by1) // 2)
    mask = 255 * np.ones(refined_crop.shape, refined_crop.dtype)
    _ = cv2.seamlessClone(refined_crop, frame, mask, center, cv2.NORMAL_CLONE)
    return time.perf_counter() - start

def benchmark_feather(frame, refined_crop, by1, by2, bx1, bx2):
    start = time.perf_counter()
    mask = np.ones((by2 - by1, bx2 - bx1), dtype=np.float32)
    mask[0:3, :] = 0; mask[-3:, :] = 0
    mask[:, 0:3] = 0; mask[:, -3:] = 0
    mask_blur = cv2.GaussianBlur(mask, (15, 15), 0)
    mask_blur = np.expand_dims(mask_blur, axis=-1)
    
    frame_crop = frame[by1:by2, bx1:bx2].astype(np.float32)
    refined_f = refined_crop.astype(np.float32)
    blended = refined_f * mask_blur + frame_crop * (1 - mask_blur)
    _ = blended.astype(np.uint8)
    return time.perf_counter() - start

def run_benchmarks():
    print("Preparing synthetic data...")
    # Synthetic 512x512 image
    frame = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    # Synthetic 120x80 mouth crop
    bx1, by1 = 200, 300
    bx2, by2 = 320, 380
    refined_crop = np.random.randint(0, 256, (by2-by1, bx2-bx1, 3), dtype=np.uint8)
    
    iters = 100
    
    print(f"Benchmarking Poisson Blending ({iters} iterations)...")
    poisson_times = [benchmark_poisson(frame.copy(), refined_crop, by1, by2, bx1, bx2) for _ in range(iters)]
    poisson_avg = sum(poisson_times) / iters
    print(f"Poisson Avg Time: {poisson_avg * 1000:.3f} ms / frame")
    
    print(f"Benchmarking Feather Blending ({iters} iterations)...")
    feather_times = [benchmark_feather(frame.copy(), refined_crop, by1, by2, bx1, bx2) for _ in range(iters)]
    feather_avg = sum(feather_times) / iters
    print(f"Feather Avg Time: {feather_avg * 1000:.3f} ms / frame")
    
    print("\nConclusion:")
    if feather_avg < poisson_avg:
        print(f"Feather is {poisson_avg / feather_avg:.2f}x faster.")
    else:
        print(f"Poisson is {feather_avg / poisson_avg:.2f}x faster.")

if __name__ == "__main__":
    run_benchmarks()
