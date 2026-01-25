import os
import pickle

path = "./eval/Cus_100/pickle/evrptw_100C_20R.pkl"
new_path = "./eval/Cus_100_sample/pickle/evrptw_100C_20R.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

new_data = []
for i, item in enumerate(data):
    if isinstance(item, dict) and ("id" in item) and (item["id"] < 100):
        new_data.append(item)

os.makedirs(os.path.dirname(new_path), exist_ok=True)

with open(new_path, "wb") as f:
    pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Loaded: {len(data)} instances")
print(f"Saved:  {len(new_data)} instances -> {new_path}")
