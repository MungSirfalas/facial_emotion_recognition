import shutil, random, os, glob
from sklearn.model_selection import train_test_split
import pandas as pd

# define folder names and destination paths
final_path = "D:/UNIR/Master_InteligenciaArtificial/2_Cuatrimestre/TFM/Develop/datasets/auxiliary/"
main_folders = ["train/", "validation/", "test/"]
folder_list = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# clear all paths
for main in main_folders:
    for folder in folder_list:
        file_names = glob.glob(final_path + main + folder + "/*.jpg")
        for file_name in file_names:
            os.remove(file_name)
print("files removed succesfully from final directory")

# Run this section for: CK+ dataset
# directory path
print("CK+ dataset start")
dir_path = "D:/UNIR/Master_InteligenciaArtificial/2_Cuatrimestre/TFM/Develop/datasets/auxiliary/ck_plus/"

# get images
folder_list_tmp = [i for i in folder_list if i != "neutral"]

for folder in folder_list_tmp:
    # shuffle data and split
    file_names = list(map(os.path.basename, glob.glob(dir_path + folder + "/*.jpg")))
    train_files, val_files = train_test_split(file_names, train_size=0.8,
                                              random_state=42, shuffle=True)
    files_dict = {"train/": train_files, "validation/": val_files}

    for main in main_folders:
        if main == "test/":
            continue
        for file_name in files_dict[main]:
            src_path = os.path.join(dir_path, folder, file_name)
            dest_path = os.path.join(final_path, main, folder, file_name)
            shutil.copyfile(src_path, dest_path)
        print("{} - {}: {}".format(folder, main, len(files_dict[main])))
print("CK+ dataset ready")

# Run this section for: modified Facial Expressions Training Data (FETD) dataset
# directory path
print("FETD dataset start")
dir_path = "D:/UNIR/Master_InteligenciaArtificial/2_Cuatrimestre/TFM/Develop/datasets/auxiliary/fetd_database/"

# get image sample
for folder in folder_list:
    # limit the number of samples for neutral class
    if folder in ["angry", "disgust", "fear"]:
        file_names = list(map(os.path.basename, glob.glob(dir_path + folder + "/*.jpg")))
        # file_names = random.sample(file_names, int(0.5*len(file_names)))
    elif folder in ["sad", "surprise"]:
        file_names = list(map(os.path.basename, glob.glob(dir_path + folder + "/*.jpg")))
        # file_names = random.sample(file_names, int(0.5*len(file_names)))
    else:
        continue
        # file_names = list(map(os.path.basename, glob.glob(dir_path + folder + "/*.jpg")))

    # shuffle data and split
    train_files, val_files = train_test_split(file_names, train_size=0.8,
                                              random_state=42, shuffle=True)
    val_files, test_files = train_test_split(val_files, train_size=0.5,
                                             random_state=42, shuffle=True)
    files_dict = {"train/": train_files, "validation/": val_files, "test/": test_files}

    for main in main_folders:
        # if main == "test/":
        #     continue
        for file_name in files_dict[main]:
            src_path = os.path.join(dir_path, folder, file_name)
            dest_path = os.path.join(final_path, main, folder, file_name)
            shutil.copyfile(src_path, dest_path)
        print("{} - {}: {}".format(folder, main, len(files_dict[main])))
print("FETD dataset ready")

# Run this section for: FER+ dataset
# directory path
print("FER+ dataset start")
dir_path = "D:/UNIR/Master_InteligenciaArtificial/2_Cuatrimestre/TFM/Develop/datasets/auxiliary/fer+/"

# copy images into folders
for main in main_folders:
    if main == "validation/":
        continue
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
                print("{} - {}: {}".format(folder, main_tmp, len(files_dict[main_tmp])))
        else:
            file_names = list(map(os.path.basename, glob.glob(dir_path + main + folder + "/*.jpg")))

            for file_name in file_names:
                src_path = os.path.join(dir_path, main, folder, file_name)
                dest_path = os.path.join(final_path, main, folder, file_name)
                shutil.copyfile(src_path, dest_path)
            print("{} - {}: {}".format(folder, main, len(file_names)))
print("FER+ dataset ready")

# Run this section for: JAFFE dataset
# directory path
print("JAFFE dataset start")
dir_path = "D:/UNIR/Master_InteligenciaArtificial/2_Cuatrimestre/TFM/Develop/datasets/auxiliary/jaffe_dataset/"

# get image sample
for folder in folder_list:
    # shuffle data and split
    file_names = list(map(os.path.basename, glob.glob(dir_path + folder + "/*.jpg")))
    train_files, val_files = train_test_split(file_names, train_size=0.8,
                                              random_state=42, shuffle=True)
    files_dict = {"train/": train_files, "validation/": val_files}

    for main in main_folders:
        if main == "test/":
            continue
        for file_name in files_dict[main]:
            src_path = os.path.join(dir_path, folder, file_name)
            dest_path = os.path.join(final_path, main, folder, file_name)
            shutil.copyfile(src_path, dest_path)
        print("{} - {}: {}".format(folder, main, len(files_dict[main])))
print("JAFFE dataset ready")

# Run this section for: Ecuadorian Facial Expressions (EFE) dataset
# take each raw image which then is copied to its corresponding label folder
# define paths
print("EFE dataset start")
dir_path = "D:/UNIR/Master_InteligenciaArtificial/2_Cuatrimestre/TFM/Develop/datasets/auxiliary/efe_dataset/0_raw_images/"
mid_path = "D:/UNIR/Master_InteligenciaArtificial/2_Cuatrimestre/TFM/Develop/datasets/auxiliary/efe_dataset/"

# clear all paths in mid path
for main in [mid_path]:
    path = main
    for folder in folder_list:
        file_names = glob.glob(path + folder + "/*.jpg")
        for file_name in file_names:
            os.remove(file_name)
print("files removed succesfully from EFE dataset directory")

# read .csv file columns: image, label
df = pd.read_csv(dir_path + "labels.csv", index_col=False)
df = df.sort_values('label')

for image, label in df.itertuples(index=False):
    src_path = os.path.join(dir_path, image)
    dest_path = os.path.join(mid_path, label, image)
    shutil.copyfile(src_path, dest_path)

# count images in Cesar Ron dataset
# print("images from Cesar Ron Dataset")
# for folder in folder_list:
#     n_images = len(list(map(os.path.basename, glob.glob(mid_path + folder + "/*.jpg"))))
#     print(folder, ":", n_images)

# get image sample
for folder in folder_list:
    # shuffle data and split
    file_names = list(map(os.path.basename, glob.glob(mid_path + folder + "/*.jpg")))
    train_files, val_files = train_test_split(file_names, train_size=0.8,
                                              random_state=42, shuffle=True)
    # val_files = test_files
    files_dict = {"train/": train_files, "validation/": val_files}

    for main in main_folders:
        if main == "test/":
            continue
        for file_name in files_dict[main]:
            src_path = os.path.join(mid_path, folder, file_name)
            dest_path = os.path.join(final_path, main, folder, file_name)
            shutil.copyfile(src_path, dest_path)
        print("{} - {}: {}".format(folder, main, len(files_dict[main])))
print("EFE dataset ready")

# Count images in final path
print("final image count")
for main in main_folders:
    for folder in folder_list:
        n_images = len(list(map(os.path.basename, glob.glob(final_path + main + folder + "/*.jpg"))))
        print(main, "-", folder, ":", n_images)