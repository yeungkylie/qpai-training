import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd

## create metrics table
path = f"I:/research\seblab\data\group_folders\Kylie/validation/metrics/"
files = glob.glob(os.path.join(path, "*.npz"))
files.sort()
collated_mae = [[] for i in range(75)]
mae_all = [[] for i in range(15)]
excel_path = os.path.join(path, "metrics.xlsx")
for file in files:
    print(file)
    data = np.load(file, allow_pickle=True)
    mae = np.array(data['all_mae']).T
    processes = data['processes']
    datasets = data['datasets']
    print(mae.shape)
    mae_all = np.append(mae_all, mae, axis=1)
    mae_col = np.reshape(mae, (75, 1))
    collated_mae = np.append(collated_mae, mae_col, axis=1)
# print(collated_mae)
print(collated_mae.shape)
df = pd.DataFrame(mae_all)
df.to_excel(excel_path, index=False)

# collated_mae_a = collated_mae []

test_datasets = ["Phantom1_flow_phantom_no_melanin",
                 "Phantom2_flow_phantom_medium_melanin",
                 "Simulation1_SingleVesselInWater",
                 "Simulation2_SingleVesselInBlood",
                 "Simulation3_VesselDeepInWater",
                 "Simulation4_HeterogeneousDistribution",
                 "Simulation5_ForearmInitialPressure",
                 "Simulation6_ForearmReconstructedData"]

# normal = plt.Normalize(collated_mae.min() - 0.01, collated_mae.max() + 0.05)
# colours = plt.cm.hot(normal(collated_mae))
# collated_mae = np.around(collated_mae, decimals=2)
# print(collated_mae)
# # fig = plt.figure(figsize=(15, 8))
# fig = plt.figure()
# ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
# the_table = plt.table(cellText=collated_mae, rowLabels=np.arange(1,71), colLabels=np.arange(1, 7),
#                       colWidths=[0.1] * collated_mae.shape[1], loc='center',
#                       cellColours=colours)
# # plt.tight_layout()
# plt.show()
