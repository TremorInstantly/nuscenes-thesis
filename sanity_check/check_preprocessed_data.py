import numpy as np

from config import PREPROCESSED_SAVE_PATH

data = np.load(PREPROCESSED_SAVE_PATH, allow_pickle=True)

print("Keys:", data.files)

for key in data.files:
    print(key, len(data[key]))
    
for i in range(10):
    print("EGO HIST:\n", data["ego_hist"][i])
    print("EGO FUT:\n", data["ego_fut"][i])
    print("NEIGHBORS:\n", data["neighbors"][i])
    print("LANES:\n", data["lanes"][i])
