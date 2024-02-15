from einops import repeat
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
y = repeat(x, 'h w -> b h w', b=3)

print(y)