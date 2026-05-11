import numpy as np

# model input shape for 1 time step
shape = (189, 64, 64)

data_t_minus_6 = np.random.randn(*shape).astype(np.float32)
data_t = np.random.randn(*shape).astype(np.float32)

np.save('data_Tmoins6h.npy', data_t_minus_6)
np.save('data_T.npy', data_t)

print("Dummy data generated: data_Tmoins6h.npy, data_T.npy")
