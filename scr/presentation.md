
### Summary of Steps in `ImprovedNet`

`Padding = Reduce total size(ex.28x28->14x14)
Stride = Lenght of step(messured on pixels)
Kernel = Size of the filter(matrix)`

1. **Conv1 (`Conv2d(1, 16, kernel_size=3, stride=1, padding=1)`)**  
   - Applies 16 filters **3x3**, `stride=1`, `padding=1`  
   - **Output:** `16x28x28` (size preserved due to `padding=1`)

2. **Conv2 (`Conv2d(16, 32, kernel_size=3, stride=1, padding=1)`)**  
   - Applies 32 filters **3x3**, `stride=1`, `padding=1`  
   - **Output:** `32x28x28`

3. **Max Pooling (`MaxPool2d(kernel_size=2, stride=2)`)**  
   - Reduces size by half, taking max values from **2x2** blocks, `stride=2`  
   - **Output:** `32x14x14`

4. **Flatten (`torch.flatten(x, 1)`)**  
   - Converts `32x14x14` into a `6272`-element vector

5. **Fully Connected Layers (`fc1`, `fc2`, `fc3`)**  
   - `fc1`: `6272 → 128`, ReLU + Dropout  
   - `fc2`: `128 → 64`, ReLU + Dropout  
   - `fc3`: `64 → 10`, final output with logits

