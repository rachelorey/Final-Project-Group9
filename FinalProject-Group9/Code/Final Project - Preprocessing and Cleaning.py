#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import numpy as np
import pandas as pd
import random

sources = {"equipment":"https://ballotpedia.org/Voting_methods_and_equipment_by_state",
          "linelength":"https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Y38VIQ",
          "demographics":"https://www.census.gov/data/tables/time-series/demo/popest/2010s-counties-detail.html#par_textimage_1383669527",
          "milage":"https://conservancy.umn.edu/handle/11299/181605",
          "state turnout":"https://worldpopulationreview.com/states/voter-turnout-by-state/"}

df = pd.read_table('https://raw.githubusercontent.com/rachelorey/Final-Project-Group9/master/MITU0022_OUTPUT.tab')
dens = pd.read_csv("https://raw.githubusercontent.com/rachelorey/Final-Project-Group9/master/county2010_hist_pops_densities.csv") 
turnout = pd.read_csv("https://raw.githubusercontent.com/rachelorey/Final-Project-Group9/master/data.csv")
dem = pd.read_csv("D:"+"\\Downloads"+'\\cc-est2018-alldata.csv',encoding='latin-1')

    
turnout = turnout[["State","totalHighestOffice","Pop"]]
turnout["turnout"]=np.array(turnout["totalHighestOffice"])/np.array(turnout["Pop"])
turnout.drop(["totalHighestOffice","Pop"],inplace=True,axis=1)
turnout.columns = ["STNAME","turnout"]

#processing demographic data
dem_clean = dem[["YEAR","STNAME","STATE","COUNTY","CTYNAME","HBAC_FEMALE","HBAC_MALE","TOT_POP","AGEGRP"]]
dem_clean = dem_clean[dem_clean["YEAR"]==11]
dem_clean = dem_clean[dem_clean["AGEGRP"]==0]
dem_clean.drop(["YEAR","AGEGRP"],axis=1,inplace=True)
dem_clean["PercentHBAC"] = (np.array(dem_clean["HBAC_FEMALE"])+np.array(dem_clean["HBAC_MALE"]))/(np.array(dem_clean["TOT_POP"]))
dem_clean["COUNTY"] = dem_clean["COUNTY"].apply(lambda x: '{0:0>3}'.format(x))
dem_clean["countyfips"] = dem_clean["STATE"].astype(str) + dem_clean["COUNTY"].astype(str)
dem_clean.drop(["HBAC_FEMALE","HBAC_MALE","STATE","COUNTY"],axis=1,inplace=True)

dens_clean = dens[["GEOID10","STATE","COUNTY","dens2010"]]

dem_clean.set_index("countyfips",inplace=True,drop=True)
dens_clean.columns = ["countyfips","STNAME","CTYNAME","dens2010"]
dens_clean.set_index("countyfips",inplace=True,drop=True)
dem_clean.index = dem_clean.index.astype('int64')

demdata = pd.merge(dem_clean,dens_clean["dens2010"],on="countyfips")

#data cleaning - equipment and turnout data
equipment = pd.read_html("https://ballotpedia.org/Voting_methods_and_equipment_by_state",match="Voting equipment usage",header=0)[0]
equipment.columns = ["STNAME","Eq Type"]
demdata_EQ = pd.merge(demdata.reset_index(),equipment,on="STNAME")
demdata_EQ = pd.merge(demdata_EQ,turnout,on="STNAME")
demdata_EQ.set_index("countyfips",drop=True,inplace=True)

#data cleaning and import line length; isolating line length and aggregating by county
linelength = pd.DataFrame(df["Q13"])
linelength["countyfips"] = df["countyfips"]
print(len(linelength))
linelength = linelength.dropna()
print(len(linelength))

#remove "I don't know" responses
linelength = linelength[linelength["Q13"]!=6]
linelength.reset_index(inplace=True,drop=True)

#bringing all the data together
full_df = pd.merge(demdata_EQ, linelength,on='countyfips') 
print(len(full_df))
full_df = full_df.dropna()
print(len(full_df))

full_df_onehot = pd.get_dummies(full_df,columns=['Eq Type'], prefix = ['EQ'])
full_df_onehot_ST = pd.get_dummies(full_df_onehot,columns=['STNAME'], prefix = ['ST_'])


# In[ ]:




