import shutil, random, os, glob
from sklearn.model_selection import train_test_split
import pandas as pd

# define folder names and destination paths
final_path = "D:/UNIR/Master_InteligenciaArtificial/2_Cuatrimestre/TFM/Develop/datasets/fer2013/"
main_folders = ["train/", "validation/", "test/"]
folder_list = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# clear all paths
for main in main_folders:
    for folder in folder_list:
        file_names = glob.glob(final_path + main + folder + "/*.jpg")
        for file_name in file_names:
            os.remove(file_name)
print("files removed succesfully from final directory")

# Run this section for: FER2013 dataset
# directory path
print("FER2013 dataset start")
dir_path = "D:/UNIR/Master_InteligenciaArtificial/2_Cuatrimestre/TFM/Develop/datasets/fer2013/fer2013/"

# copy images into folders
for main in [i for i in main_folders if i != "validation/"]:
    # get images
    for folder in folder_list:
        if main == "test/":
            file_names = list(map(os.path.basename, glob.glob(dir_path + main + folder + "/*.jpg")))
            val_files, test_files = train_test_split(file_names, train_size=0.5, test_size=0.5,
                                                     random_state=42, shuffle=True)
            files_dict = {"validation/": val_files, "test/": test_files}

            for main_tmp in [i for i in main_folders if i != "train/"]:
                for file_name in files_dict[main_tmp]:
                    src_path = os.path.join(dir_path, main, folder, file_name)
                    dest_path = os.path.join(final_path, main_tmp, folder, file_name)
                    shutil.copyfile(src_path, dest_path)
        else:
            file_names = list(map(os.path.basename, glob.glob(dir_path + main + folder + "/*.jpg")))

            for file_name in file_names:
                src_path = os.path.join(dir_path, main, folder, file_name)
                dest_path = os.path.join(final_path, main, folder, file_name)
                shutil.copyfile(src_path, dest_path)
print("FER2013 dataset ready")

# Count images in final path
print("final image count")
for main in main_folders:
    for folder in folder_list:
        n_images = len(list(map(os.path.basename, glob.glob(final_path + main + folder + "/*.jpg"))))
        print(main, "-", folder, ":", n_images)