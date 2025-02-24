import numpy as np
import time

# Create two large random arrays
size = 10**6  # One million elements
array1 = np.random.rand(size)
array2 = np.random.rand(size)

# Method 1: Using a loop (non-vectorized)
start_time = time.time()
result_loop = [array1[i] + array2[i] for i in range(size)] + [array1[i] + array2[i] for i in range(size)]
loop_time = time.time() - start_time

# Method 2: Using NumPy vectorized operation
start_time = time.time()
result_vectorized = array1 + array2  + array1 + array2 # Element-wise addition
vectorized_time = time.time() - start_time

# Display results
print(f"Time taken using loop: {loop_time:.6f} seconds")
print(f"Time taken using NumPy vectorization: {vectorized_time:.6f} seconds")
print(f"Speedup: {loop_time / vectorized_time:.2f}x faster")

