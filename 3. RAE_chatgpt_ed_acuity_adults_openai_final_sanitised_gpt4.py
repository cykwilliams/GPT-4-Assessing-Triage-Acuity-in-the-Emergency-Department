import pandas as pd
import openai
import os
import re
import json
import base64
import datetime
import requests
import urllib.parse
from dotenv import load_dotenv

from ratelimit import limits, sleep_and_retry

df_testB_output = pd.read_csv('path/ed_notes_edprovider_adults_master_processed_df_testB_output.csv', index_col = 0)

load_dotenv('.env')
API_KEY = os.environ.get('STAGE_API_KEY')
API_VERSION = os.environ.get('API_VERSION')
RESOURCE_ENDPOINT = os.environ.get('RESOURCE_ENDPOINT')

openai.api_type = "azure"  
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT  
openai.api_version = '2023-03-15-preview'

##Select GPT deployment (3.5 vs 4)
#deployment_name='gpt-35-turbo'
deployment_name='gpt-4'

# Define the rate limit for the function (e.g. 35 calls per second)
@sleep_and_retry
@limits(calls=295, period=60)
def run_chatgpt_api(prompt):
    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages = [
                {"role": "user", "content": prompt}
            ],
            n=1,
            stop=None,
            temperature=0,
            )
    except:
        response = 'Error_with_API_CYKW'
    return response

def retrieve_content_from_response_json2(x):
    try:
        return json.loads(str(x))['choices'][0]['message']['content']
    except:
        return 'Error_with_API_CYKW'

def retrieve_label(x):
    if '0' in x:
        label = '0'
    elif '1' in x:
        label = '1'
    elif '0' in x and '1' in x:
        label = 'both_present'
    elif 'Error_with_API_CYKW' in x:
        label = 'error'
    else:
        label = 'neither'
    
    return label
    
def process_chatgpt_output_acuity_ed(df):
    print('Saving temp df')
    import csv
    #Save temp df:
    df.to_csv('temp_chatgpt_output.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')
    print('retrieving content')
    df['response_content'] =  df['response_json'].apply(lambda x: retrieve_content_from_response_json2(x))
    print('retrieving label')
    df['label'] = df['response_content'].apply(lambda x: retrieve_label(x))
    print('Saving temp df2')
    #Save temp df2
    df.to_csv('temp_chatgpt_output2' + '.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')
    return df

def combined_lists_to_df_acuity_ed(df, encounter_id_list, prompt_list, response_list):
    output_df = pd.DataFrame([encounter_id_list, prompt_list, response_list]).transpose()
    output_df = output_df.rename(columns = {0:'visit_occurrence_id_combined', 1:'prompt', 2:'response_json'})
    df = df.merge(output_df, left_on = ['visit_occurrence_id_combined', 'prompt'], right_on = ['visit_occurrence_id_combined', 'prompt'], how = 'left')
    df = df[df['response_json'].notnull()]
    df = process_chatgpt_output_acuity_ed(df)
    return df 


df_to_run = df_testB_output.copy()
print('Anticipated cost:', df_to_run.prompt_tokens.sum()*0.06/1000)

encounter_id_list = []
prompt_list = []
response_list = []

for key, value in dict(zip(df_to_run['visit_occurrence_id_combined'].tolist(), df_to_run['prompt'].tolist())).items():
    print(key)
    print(len(response_list))
    encounter_id_list.append(key)
    prompt_list.append(value)
    response_list.append(run_chatgpt_api(value))

df_to_save = combined_lists_to_df_acuity_ed(df_to_run, encounter_id_list, prompt_list, response_list)

import csv
df_to_save.to_csv("path/ed_notes_edprovider_adults_master_processed_df_testB_output_results_GPT4.csv", header=True,
      quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')

#Do the same with GPT-3.5 (using above code)

df_to_save['prompt_tokens'] = df_to_save['response_json'].apply(lambda x: x['usage']['prompt_tokens'])
df_to_save['completion_tokens'] = df_to_save['response_json'].apply(lambda x: x['usage']['completion_tokens'])


# ## Analysis

# ### Full n=10000 sample

# #### GPT-4

import pandas as pd
df_gpt4_10000_results = pd.read_csv('path/ed_notes_edprovider_adults_master_processed_df_testB_output_results_GPT4.csv', index_col = 0)


#Check if the outputs are as specified
print('Count number of label and higher_acuity_patient values')
print(df_gpt4_10000_results['label'].value_counts())
print(df_gpt4_10000_results['higher_acuity_patient'].value_counts())

#Change label from 0 (for A) and 1 (for B) to A/B:
df_gpt4_10000_results['label'] = df_gpt4_10000_results['label'].replace({0:'A', 1:'B'})


print('\n Count number of correct labels from ChatGPT')
df_gpt4_10000_results['correct_label'] = 0
df_gpt4_10000_results.loc[df_gpt4_10000_results['label'] == df_gpt4_10000_results['higher_acuity_patient'], 'correct_label'] = 1
print(df_gpt4_10000_results['correct_label'].value_counts())


##Calculate accuracy and F1 scores
from sklearn.metrics import accuracy_score, f1_score
print('Number of correct labels:', (df_gpt4_10000_results['higher_acuity_patient'] == df_gpt4_10000_results['label']).sum())
accuracy = accuracy_score(df_gpt4_10000_results['higher_acuity_patient'], df_gpt4_10000_results['label'])
f1 = f1_score(df_gpt4_10000_results['higher_acuity_patient'], df_gpt4_10000_results['label'], pos_label='A')

print('Accuracy (overall) = ', accuracy)
print('F1 score (overall) = ', f1, '\n\n')

#Categorise the groups of pairs:
acuity_pairs_map = {
    'Immediate_Emergent':0, 'Emergent_Immediate':0,
    'Immediate_Urgent':1, 'Urgent_Immediate':1,
    'Immediate_Less Urgent':2, 'Less Urgent_Immediate':2,
    'Immediate_Non-Urgent':3, 'Non-Urgent_Immediate':3,
    'Emergent_Urgent':4, 'Urgent_Emergent':4,
    'Emergent_Less Urgent':5, 'Less Urgent_Emergent':5,
    'Emergent_Non-Urgent':6, 'Non-Urgent_Emergent':6,
    'Urgent_Less Urgent':7, 'Less Urgent_Urgent':7,
    'Urgent_Non-Urgent':8, 'Non-Urgent_Urgent':8,
    'Less Urgent_Non-Urgent':9, 'Non-Urgent_Less Urgent':9                   
}

df_gpt4_10000_results['acuity_pair_class'] = df_gpt4_10000_results['acuitylevel_combined'].map(acuity_pairs_map)

#Examine accuracy/F1 score by acuity_pair_class
def stratify_by_distance(df):
    for i in sorted(df['acuity_pair_class'].unique().tolist()):
        sample = df[df['acuity_pair_class'] == i]
        accuracy = accuracy_score(sample['higher_acuity_patient'], sample['label'])
        f1 = f1_score(sample['higher_acuity_patient'], sample['label'], pos_label='A')
        print('acuity_pair_class = ', i)
        print('Number of pairs:', len(sample))
        print('value_counts of ground truth label:', sample['higher_acuity_patient'].value_counts())
        print('Accuracy = ', accuracy)
        print('F1 score = ', f1, '\n')
        

stratify_by_distance(df_gpt4_10000_results)


# #### GPT-3.5
df_gpt35_10000_results = pd.read_csv('path/ed_notes_edprovider_adults_master_processed_df_testB_output_results_GPT35.csv', index_col = 0)

#Confirm ids match between gpt4 and gpt35 result dfs
print((df_gpt4_10000_results[['visit_occurrence_id_combined']].merge(df_gpt35_10000_results[['visit_occurrence_id_combined']], how = 'inner', on = 'visit_occurrence_id_combined')).shape)
#(10000, 1) - hence they do

#Check if the outputs are as specified
print('Count number of label and higher_acuity_patient values')
print(df_gpt35_10000_results['label'].value_counts())
print(df_gpt35_10000_results['higher_acuity_patient'].value_counts())

#Change label from 0 (for A) and 1 (for B) to A/B:
df_gpt35_10000_results['label'] = df_gpt35_10000_results['label'].replace({0:'A', 1:'B'})

print('\n Count number of correct labels from ChatGPT')
df_gpt35_10000_results['correct_label'] = 0
df_gpt35_10000_results.loc[df_gpt35_10000_results['label'] == df_gpt35_10000_results['higher_acuity_patient'], 'correct_label'] = 1
print(df_gpt35_10000_results['correct_label'].value_counts())

##Calculate accuracy and F1 scores
from sklearn.metrics import accuracy_score, f1_score
print('Number of correct labels:', (df_gpt35_10000_results['higher_acuity_patient'] == df_gpt35_10000_results['label']).sum())
accuracy = accuracy_score(df_gpt35_10000_results['higher_acuity_patient'], df_gpt35_10000_results['label'])
f1 = f1_score(df_gpt35_10000_results['higher_acuity_patient'], df_gpt35_10000_results['label'], pos_label='A')

print('Accuracy (overall) = ', accuracy)
print('F1 score (overall) = ', f1, '\n\n')

#Categorise the groups of pairs:
acuity_pairs_map = {
    'Immediate_Emergent':0, 'Emergent_Immediate':0,
    'Immediate_Urgent':1, 'Urgent_Immediate':1,
    'Immediate_Less Urgent':2, 'Less Urgent_Immediate':2,
    'Immediate_Non-Urgent':3, 'Non-Urgent_Immediate':3,
    'Emergent_Urgent':4, 'Urgent_Emergent':4,
    'Emergent_Less Urgent':5, 'Less Urgent_Emergent':5,
    'Emergent_Non-Urgent':6, 'Non-Urgent_Emergent':6,
    'Urgent_Less Urgent':7, 'Less Urgent_Urgent':7,
    'Urgent_Non-Urgent':8, 'Non-Urgent_Urgent':8,
    'Less Urgent_Non-Urgent':9, 'Non-Urgent_Less Urgent':9                   
}

df_gpt35_10000_results['acuity_pair_class'] = df_gpt35_10000_results['acuitylevel_combined'].map(acuity_pairs_map)

#Examine accuracy/F1 score by acuity_pair_class
def stratify_by_distance(df):
    for i in sorted(df['acuity_pair_class'].unique().tolist()):
        sample = df[df['acuity_pair_class'] == i]
        accuracy = accuracy_score(sample['higher_acuity_patient'], sample['label'])
        f1 = f1_score(sample['higher_acuity_patient'], sample['label'], pos_label='A')
        print('acuity_pair_class = ', i)
        print('Number of pairs:', len(sample))
        print('value_counts of ground truth label:', sample['higher_acuity_patient'].value_counts())
        print('Accuracy = ', accuracy)
        print('F1 score = ', f1, '\n')
        

stratify_by_distance(df_gpt35_10000_results)


# ### CYKW n=500 sample
###Examine manually classified subsample
df_gpt35_500_results = pd.read_csv('path\Manual_annotation\df_CYKW_500_results.csv', index_col = 0)
##Note, this contains the labels from the GPT-3.5 analysis (annotation was done in a blinded manner then, once complete, manually added as columns imto df_CYKW_500_results.csv)

# #### GPT-3.5
from sklearn.metrics import accuracy_score, f1_score

# Assuming model_output and ground_truth are numpy arrays or lists of the same length
accuracy_df_gpt35_500_results = accuracy_score(df_gpt35_500_results['higher_acuity_patient'], df_gpt35_500_results['label'])
f1_df_gpt35_500_results = f1_score(df_gpt35_500_results['higher_acuity_patient'], df_gpt35_500_results['label'], pos_label='A')

print('Accuracy = (ChatGPT)', accuracy_df_gpt35_500_results)
print('F1 score = (ChatGPT)', f1_df_gpt35_500_results)

accuracy_df_gpt35_500_results_cykw = accuracy_score(df_gpt35_500_results['higher_acuity_patient'], df_gpt35_500_results['CYKW_label_binary'])
f1_df_gpt35_500_results_cykw = f1_score(df_gpt35_500_results['higher_acuity_patient'], df_gpt35_500_results['CYKW_label_binary'], pos_label='A')

print('Accuracy (CYKW) = ', accuracy_df_gpt35_500_results_cykw)
print('F1 score (CYKW) = ', f1_df_gpt35_500_results_cykw)

def stratify_by_distance(df):
    for i in sorted(df['acuity_pair_class'].unique().tolist()):
        sample = df[df['acuity_pair_class'] == i]
        accuracy = accuracy_score(sample['higher_acuity_patient'], sample['label'])
        f1 = f1_score(sample['higher_acuity_patient'], sample['label'], pos_label='A')
        print('acuity_pair_class = ', i)
        print('Number of pairs:', len(sample))
        print('value_counts of ground truth label:', sample['higher_acuity_patient'].value_counts())
        print('Accuracy = ', accuracy)
        print('F1 score = ', f1)
        print('Number correct = ', 50*accuracy, '\n')
        
stratify_by_distance(df_gpt35_500_results)

# #### GPT-4
##Add the df_gpt35_500_results CYKW_labels to the GPT-4 labels
#Need to add CYKW_label from df_gpt35_500_results to the same visit_occurrence_id_combined from df_gpt4_10000_results, for only the n = 500 samples
df_gpt4_500_results = df_gpt4_10000_results[df_gpt4_10000_results['visit_occurrence_id_combined'].isin(df_gpt35_500_results.visit_occurrence_id_combined)]
print(df_gpt4_500_results.shape)
print(df_gpt4_500_results.shape)

df_gpt4_500_results = df_gpt4_500_results.merge(df_gpt35_500_results[['visit_occurrence_id_combined', 'CYKW_label', 'CYKW_label_binary']], on = 'visit_occurrence_id_combined', how = 'left')
print(df_gpt4_500_results.shape)

print(df_gpt4_500_results.shape)
print(df_gpt4_500_results.acuity_pair_class.value_counts())

# Assuming model_output and ground_truth are numpy arrays or lists of the same length
accuracy_df_gpt4_500_results = accuracy_score(df_gpt4_500_results['higher_acuity_patient'], df_gpt4_500_results['label'])
f1_df_gpt4_500_results = f1_score(df_gpt4_500_results['higher_acuity_patient'], df_gpt4_500_results['label'], pos_label='A')

print('Accuracy = (ChatGPT)', accuracy_df_gpt4_500_results)
print('F1 score = (ChatGPT)', f1_df_gpt4_500_results)

accuracy_df_gpt4_500_results_cykw = accuracy_score(df_gpt4_500_results['higher_acuity_patient'], df_gpt4_500_results['CYKW_label_binary'])
f1_df_gpt4_500_results_cykw = f1_score(df_gpt4_500_results['higher_acuity_patient'], df_gpt4_500_results['CYKW_label_binary'], pos_label='A')

print('Accuracy (CYKW) = ', accuracy_df_gpt4_500_results_cykw)
print('F1 score (CYKW) = ', f1_df_gpt4_500_results_cykw)

def stratify_by_distance(df):
    for i in sorted(df['acuity_pair_class'].unique().tolist()):
        sample = df[df['acuity_pair_class'] == i]
        accuracy = accuracy_score(sample['higher_acuity_patient'], sample['label'])
        f1 = f1_score(sample['higher_acuity_patient'], sample['label'], pos_label='A')
        print('acuity_pair_class = ', i)
        print('Number of pairs:', len(sample))
        print('value_counts of ground truth label:', sample['higher_acuity_patient'].value_counts())
        print('Accuracy = ', accuracy)
        print('F1 score = ', f1)
        print('Number correct = ', 50*accuracy, '\n')
        

stratify_by_distance(df_gpt4_500_results)

def stratify_by_distance_cykw(df):
    for i in sorted(df['acuity_pair_class'].unique().tolist()):
        sample = df[df['acuity_pair_class'] == i]
        accuracy = accuracy_score(sample['higher_acuity_patient'], sample['CYKW_label_binary'])
        f1 = f1_score(sample['higher_acuity_patient'], sample['CYKW_label_binary'], pos_label='A')
        print('acuity_pair_class = ', i)
        print('Number of pairs:', len(sample))
        print('value_counts of ground truth label:', sample['higher_acuity_patient'].value_counts())
        print('Accuracy = ', accuracy)
        print('F1 score = ', f1)
        print('Number correct = ', 50*accuracy, '\n')
        
stratify_by_distance_cykw(df_gpt4_500_results)
