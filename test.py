from Utils.eval import *

# eval_results("/Users/jz/Code/data/Warped_pairs",
#              ref_img_path="/Users/jz/Code/data/results/transformed",
#              ref_label_path="/Users/jz/Code/data/results/transformed_labels",
#              output_path="/Users/jz/Code/data/Warped_pairs/")

import json

file = "/Users/jz/Code/data/Revised_data/registration_affine/label_results.json"
with open(file, 'r') as f:
   label_results: dict = json.load(f)
print(f"Loading {file}")

assd_list = list()
dsc_list = list()
verror_list = list()

for k, v in label_results.items():
    assd_list.append(
        [float(assd) for assd in v["assd"]]
    )
    dsc_list.append(
        [float(dsc) for dsc in v["dsc"]]
    )
    verror_list.append(
        [float(verror) for verror in v["verror"]]
    )

mean_assd = np.mean(np.array(assd_list), axis=0)
mean_dsc = np.mean(np.array(dsc_list), axis=0)
mean_verror = np.mean(np.array(verror_list), axis=0)

for i in range(1, 12):
    print(f"The average DSC for class {i} is {mean_dsc[i]}")
    print(f"The average ASSD for class {i} is {mean_assd[i]}")
    print(f"The average VError for class {i} is {mean_verror[i]}")
    print()

print(f"The overall average ASSD is {np.mean(mean_assd)}")
print(f"The overall average DSC is {np.mean(mean_dsc)}")
print(f"The overall average VError is {np.mean(mean_verror)}")
