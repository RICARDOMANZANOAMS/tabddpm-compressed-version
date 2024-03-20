import numpy as np
import pandas as pd
def read_file(path):
    a=np.load(path)
    b=pd.DataFrame(a)
    print(b)
    return b

file_path='C:/RICARDO/2023 CISTECH BACKUP/cistech/2024/IoT/Diffusion/Tab-ddpm/tab-ddpm/data/data/churn2 test/X_num_train.npy'
print("train dataset")
train_features=read_file(file_path)
file_path='C:/RICARDO/2023 CISTECH BACKUP/cistech/2024/IoT/Diffusion/Tab-ddpm/tab-ddpm/data/data/churn2 test/y_train.npy'
print("train label dataset")
train_labels=read_file(file_path)



#test 
file_path='C:/RICARDO/2023 CISTECH BACKUP/cistech/2024/IoT/Diffusion/Tab-ddpm/tab-ddpm/data/data/churn2 test/results/X_num_train.npy'
print("generated train dataset")
generated_features=read_file(file_path)
file_path='C:/RICARDO/2023 CISTECH BACKUP/cistech/2024/IoT/Diffusion/Tab-ddpm/tab-ddpm/data/data/churn2 test/results/y_train.npy'
print("generated label train dataset")
generated_labels=read_file(file_path)

# print("check ")
# for index, row in generated_features.iterrows():
#     #feature_0=row[0]
#     feature_1=row[1]
#     # feature_2=row[2]
#     # feature_3=row[3]
#     # feature_4=row[4]
#     # feature_5=row[5]
#     # feature_6=row[6]
   


#     for index_gen, row_gen in train_features.iterrows():
#         feature_0_gen=row_gen [0]
#         feature_1_gen=row_gen [1]
#         feature_2_gen=row_gen [2]
#         feature_3_gen=row_gen [3]
#         feature_4_gen=row_gen [4]
#         feature_5_gen=row_gen [5]
#         feature_6_gen=row_gen [6]
        
#         # if feature_0==feature_0_gen:
#         #     print("there is a value equal")
#         #     print(row)
#         if feature_1==feature_1_gen:
#             print("there is a value equal")
#             print(row)
#         # if feature_2==feature_2_gen:
#         #     print("there is a value equal")
#         #     print(row)
#         # if feature_3==feature_3_gen:
#         #     print("there is a value equal")
#         #     print(row)
#         # if feature_4==feature_4_gen:
#         #     print("there is a value equal")
#         #     print(row)
#         # if feature_5==feature_5_gen:
#         #     print("there is a value equal")
#         #     print(row)
#         # if feature_6==feature_6_gen:
#         #     print("there is a value equal")
#         #     print(row)
#     print("Next")

   

