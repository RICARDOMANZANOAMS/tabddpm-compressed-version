from wrapper import wrapper
import pandas as pd
path='C:/RICARDO/CISTECH-CISTEL/cistech/2024/IoT/Diffusion/Tab-ddpm/Tab-ddpm repo/tab-ddpm/data_generation_conf_results/UNSW_2018_IoT_Botnet_Full5pc_1.csv'
df=pd.read_csv(path)
print(df)

# theft_rows=df[df['category']=='Theft']
# normal_rows=df[df['category']=='Normal']
# df=pd.concat([theft_rows,normal_rows])
# print("filered dataset")
# print(df)
print(df['category'].value_counts())
features=['stime','Pkts_P_State_P_Protocol_P_DestIP']
label='category'
test_size=0.2
parent_dir='C:/RICARDO/CISTECH-CISTEL/cistech/2024/IoT/Diffusion/Tab-ddpm/Tab-ddpm repo/tab-ddpm/data_generation_conf_results/results'
real_data_path='C:/RICARDO/CISTECH-CISTEL/cistech/2024/IoT/Diffusion/Tab-ddpm/Tab-ddpm repo/tab-ddpm/data_generation_conf_results'
wrapper(df,features, label, test_size,parent_dir,real_data_path)
        
        
        