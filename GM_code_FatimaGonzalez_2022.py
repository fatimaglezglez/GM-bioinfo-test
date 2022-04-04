#!/usr/bin/env python
# coding: utf-8

# Prueba para puesto de Bioinformática en el Gregorio Marañón
# Fátima González González - Abril 2022

# imports 
import os
import pandas as pd 
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist, squareform
import numpy as np
from matplotlib import pyplot as plt

# function definitions

# STEP 1: create a function to read a single VCF file
def read_vcf(vcf_file):
    with open(vcf_file, 'r') as f:
        data = []
        no_names=True
        for line in f:
            if line[0]=='#':
                last_line=line.strip('#')
                continue
            while no_names:
                names_of_fields = last_line.strip().split() #safe because the file is always going to start with at least one row beginning with '#'
                no_names=False
                df = pd.DataFrame(data, columns=names_of_fields)
            line = line.strip().split()
            sline = pd.Series(line, index = names_of_fields)
            df = df.append(sline, ignore_index=True) # add row to data frame
    return df

# STEP 2: extract relevant information from parsed VCF
# filter SNPs. M. tuberculosis is an haploid organism.

#dropping heterozygous and homozygous ref/ref
def drop_gt0(df_vcf):
    gt_0 = df_vcf.index[df_vcf[df_vcf.columns[-1]].astype(str).str[0] == '0'].tolist()      # same as searching for AF 0.5 in INFO for heterozygous
    df = df_vcf.drop(gt_0)
    print('Found '+ str(len(gt_0))+' heterozygous or homozygous ref/ref that have been dropped') #Therefore:\n'+'Homozygous\t'+str(len(df_vcf.index.tolist())))
    df = df.reset_index(drop=True)
    return df

# filtering variants by quality > 20 (all, but good to have a function to adjust this if necessary)
def quality(df_vcf):
    df_vcf['QUAL'] = df_vcf['QUAL'].astype(float)
    low_qual = df_vcf.index[df_vcf['QUAL'] < 20].tolist()     
    df = df_vcf.drop(low_qual)
    df = df.reset_index(drop=True) 
    print('Found '+ str(len(low_qual))+' low quality SNPs that have been dropped')
    return df

# extracting relevant info
def extract_info(df_vcf):
    sample_name = df_vcf.columns[-1]
    df = df_vcf.reset_index(drop=True)  
    an = []
    for index, row in df.iterrows():
        info = [data.split('=') for data in row['INFO'].split(';')]
        an.append(float(info[2][1]))
    df = df_vcf.drop(columns=['CHROM', 'ID', 'QUAL', 'FILTER', 'INFO', 'FORMAT', sample_name]) 
    df['AN'] = an
    df['POS'] = df['POS'].astype(float)
    df['AN'] = df['AN'].astype(float)
    df[sample_name] = 1
    return df

# knowing if we have indels and different alt alleles for the same SNP
def any_indels(df):
    ref_alt = True
    ref_alt_pos = []
    in_dels = True
    in_dels_pos = []
    for i,row in df.iterrows(): 
        if row['AN'] != 2:
            print('Different REF/ALT:\tPOS:', row['POS'],'\tREF:', row['REF'],'\tALT:', row['ALT'])
            ref_alt_pos.append(row['POS'])
            ref_alt = False
        if (len(row['REF']) != 1)  or (len(row['ALT']) != 1):
            print('There are INDELS:\tPOS:', row['POS'],'\tREF:', row['REF'],'\tALT:', row['ALT'], '\tAN:', row['AN'])
            in_dels = False
            in_dels_pos.append(row['POS'])
        if ('+' in row['REF']) or ('+' in row['ALT']):
            print('Different REF/ALT or INDELS:\tPOS:', row['POS'],'\tREF:', row['REF'],'\tALT:', row['ALT'])
    if ref_alt:
        print('All SNPs have the same REF/ALT')
    if in_dels:
        print('There are no INDELS')
    return
# there are no differences in alt alleles for the same SNP in this data so we will not need to add anything else to the ditances calculation

# STEP 3: combine present SNP into a precence matrix
def join_data(df, df2):
    # merge
    #in_dels = []
    #alt = []
    df=df.merge(df2, how='outer', on=['POS']) # there is no SNP ID so we'll use POS to identify each SNP
    if df.isnull().values.any():  # to make sure there are some NaN, meaning the two samples are different 
        #NaN will be placed whenever the SNPs don't match between the two samples. SNP can be point mutation or INDEL.
        for i,row in df.iterrows(): 
            # finding different REF/ALT to not loose that information
            
            # commented out this section because all SNPs present in the sample files have the same REF/ALT, even INDELS
            #if not row.isnull().any(): #avoid rows with nan values (snp not presen in one sample)
            #    if (row['REF_x'] != row['REF_y']): # there is a different ALT // DO WITH REFF!!! FOR INDELS!!!!!!!!!
            #        if (len(row['REF_x']) != 1)  or (len(row['REF_y']) != 1):
                            #in_dels.append([row['POS'],  df.columns[-2], df.columns[-1]])
            #                df.at[i, 'REF_x'] = row['REF_x']+'+'+row['REF_y']
            #        elif (row['REF_y'] not in row['REF_x'].split('+')):
            #                df.at[i, 'REF_x'] = row['REF_x']+'+'+row['REF_y']   
            #    if (row['ALT_x'] != row['ALT_y']): # there is a different ALT // DO WITH REFF!!! FOR INDELS!!!!!!!!!
            #        if (len(row['ALT_x']) != 1)  or (len(row['ALT_y']) != 1):
            #                #in_dels.append([row['POS'],  df.columns[-2], df.columns[-1]])
            #                df.at[i, 'ALT_x'] = row['ALT_x']+'+'+row['ALT_y']
            #        elif (row['ALT_y'] not in row['ALT_x'].split('+')):
                            #alt.append([row['POS'],  df.columns[-2], df.columns[-1]])
            #                df.at[i, 'ALT_x'] = row['ALT_x']+'+'+row['ALT_y']               
            #                df.at[i, 'AN_x'] += 1
   
            # relocating info when SNP in df2 to X columns
            if pd.isnull(row['REF_x']):
                df.at[i, 'REF_x'] = row['REF_y']
                df.at[i, 'ALT_x'] = row['ALT_y']
                df.at[i, 'AN_x'] = row['AN_y']
    # remove unnecessary columns and rename
    df= df.drop(columns =['REF_y','ALT_y', 'AN_y'])
    df.rename(columns = {'REF_x':'REF','ALT_x':'ALT', 'AN_x':'AN'}, inplace = True)
    return df

# STEP 4: determine SNP distances between all samples
# STEP 5 (BONUS): Include INDELS
# pairwise discances between each pair of samples (manually)
def distance(df):
    d = df.isnull().sum().sum() #does not consider different ALT at same POS, does not consider INDELS
    return d


# run

#read and extract info to get presence matrix
for i,file in enumerate(os.listdir('data/')):
    name = file.split('.')[0]
    print(name)
    df = read_vcf('data/'+file)
    df = drop_gt0(df)
    df = quality(df)
    df = extract_info(df)
    globals()[f'df_{name}'] = df 
    if i == 0:
        merged_df = df
    else:
        merged_df = join_data(merged_df, df)
print('Binary Presence Matrix: \n', merged_df.fillna(0))
any_indels(merged_df)

# pairwise distances calculation
names = [f.split('.')[0] for f in os.listdir('data/')]
m = np.zeros((len(os.listdir('data/')), len(os.listdir('data/'))))
for i in range(len(os.listdir('data/'))):
    for j in range(i, len(os.listdir('data/'))):
        m_df = join_data(globals()[f'df_{names[i]}'],globals()[f'df_{names[j]}'])
        d = distance(m_df)
        m[i][j] = d
# I didn't use the stored merged dataframe because
# I would have to compare all pair of sample columns to check for similarities row by row
# whereas this way the compute of differences between samples is so easy and fast by counting Nulls

# STEP 6 (BONUS): Represent distance in a pylogenetic tree
# linkage and dendogram 
X = pdist (m, 'euclidean')
Z = linkage(X, 'ward')
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(25, 10))
plt.title('Dendogram M. tuberculosis isolates', fontsize=25)
dn = dendrogram(Z,labels=names, orientation="left")
fig.savefig('dendogram_FG.png', bbox_inches="tight")
print('Linkage and dendogram computed. Dendogram saved as dendogram_FG.png \nPlease, close shown image to get the code running again.')
plt.show()
