from Utils.utils import *

data = sorted(glob.glob(os.path.join("/Users/jz/Code/data/Warped_pairs/", "*.nii.gz")))
img_list = [os.path.basename(img) for img in data if "mdixon_r.nii.gz" in img]
label_list = [os.path.basename(label) for label in data if "segmentation_r.nii.gz" in label]

img_path = "/Users/jz/Code/data/results/transformed"
label_path = "/Users/jz/Code/data/results/transformed_labels"

img_list = [os.path.join(img_path, img) for img in img_list]
label_list = [os.path.join(label_path, label) for label in label_list]

template_img = "/Users/jz/Code/data/template.nii.gz"
template_label = "/Users/jz/Code/data/template_view.nii.gz"

output_path = "/Users/jz/Code/data/registered_affine"
output_path_affine_file = f"{output_path}/affine_file"
make_dir_if_none(output_path)
make_dir_if_none(output_path_affine_file)

for img in img_list:
    command = ["mrregister", "-type", "affine", "-transformed", f"{os.path.join(output_path, os.path.basename(img))}",
               "-affine", os.path.join(output_path_affine_file, os.path.basename(img).replace(".nii.gz",
                                                                                              ".txt")),
               "-force", template_img, img]
    subprocess.run(command)