import os
import openai
import tiktoken
import pandas as pd
import re
import csv
import random
import numpy as np
import json
from ratelimit import limits, sleep_and_retry
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_parquet('path/ed_notes_edprovider_adults_master.parquet')
df['visit_occurrence_id'] = df['visit_occurrence_id'].astype(str)
df['note_text'] = df['note_text'].astype(str)

print(df.shape)

##Conduct minimal preprocessing
#Remove '\n' (as upon inspection these appear to be randomly inserted into text)
df["note_text_processed"] = [re.sub(r'\n', '', s) for s in df["note_text"]]
#Confirm this
print(df[df['note_text_processed'].str.contains('\n')].shape, '- should be 0')

#Remove extra spaces
def remove_extra_spaces(text):
    # Use regular expressions to replace multiple spaces with a single space
    return re.sub(' +', ' ', text)

df['note_text_processed'] = df['note_text_processed'].apply(remove_extra_spaces)

#Check for duplicate encounters (this represents >1 note available for encounter due to merge)
print(len(df[df['visit_occurrence_id'].duplicated(keep = False)]))

#Sort by note time and then length per above
df['deid_service_date'] = pd.to_datetime(df['deid_service_date'])
df['note_length_words'] = df['note_text_processed'].str.len()
df = df.sort_values(['visit_occurrence_id', 'deid_service_date', 'note_length_words'], ascending=[True, True, False])

#Drop duplicates (i.e drop notes wth >1 note available, keeping the first note / the longest note in instances of two notes with the same charttime)
df = df.drop_duplicates(subset = 'visit_occurrence_id', keep = 'first')

#Examine counts of various section headers within the ED note
df['history_chiefcomplaint'] = df['note_text_processed'].str.contains('History Chief Complaint').apply(lambda x: 'Y' if x else 'N')
df['chiefcomplaint'] = df['note_text_processed'].str.contains('Chief Complaint').apply(lambda x: 'Y' if x else 'N')
df['systemsreview'] = df['note_text_processed'].str.contains('Review of Systems').apply(lambda x: 'Y' if x else 'N')
df['physicalexam'] = df['note_text_processed'].str.contains('Physical Exam').apply(lambda x: 'Y' if x else 'N')
df['edcourse'] = df['note_text_processed'].str.contains('ED Course').apply(lambda x: 'Y' if x else 'N')
df['initialassessment'] = df['note_text_processed'].str.contains('Initial Assessment').apply(lambda x: 'Y' if x else 'N')
df['plan'] = df['note_text_processed'].str.contains('Plan').apply(lambda x: 'Y' if x else 'N')
df['plan2'] = df['note_text_processed'].str.contains('Plan:').apply(lambda x: 'Y' if x else 'N')

for column in ['history_chiefcomplaint', 'chiefcomplaint', 'systemsreview',
       'physicalexam', 'edcourse', 'initialassessment', 'plan', 'plan2']:
    print('\n', column)
    print(df[column].value_counts())


def extract_text(text, start_pattern, end_pattern):
    start_regex = re.compile('|'.join(start_pattern))
    end_regex = re.compile('|'.join(end_pattern))

    try:
        start_match = start_regex.search(text) 
        end_match = end_regex.search(text) 
        start = start_match.start()
        end = end_match.start()
        result = text[start:end]
    except AttributeError:
        result = 'unable_to_extract'
    return result


def extract_initialassessment_to_end(text, initialassessment, edcourse):
    #Search first for 'Initial Assessment' and if present select text from there to end
    #Otherwise search for 'ED Course' and do the same
    #Otherwise return 'unable_to_extract'
    initialassessment_regex = re.compile('|'.join(initialassessment))
    edcourse_regex = re.compile('|'.join(edcourse))
    
    start_match = initialassessment_regex.search(text)
    if start_match is None:
        start_match = edcourse_regex.search(text)
    if start_match is None:
        return 'unable_to_extract'
    start = start_match.start()
    return text[start:]


# Apply the function and create new columns for each section

#Create list of upper/lower case variations of desired note heading
#Note that e.g all lowercase 'initial assessment' has several false positives, so settle for only 1) First letter caps and 2) all caps
chiefcomplaint = ['Chief Complaint', 'CHIEF COMPLAINT']
physicalexam = ['Physical Exam', 'PHYSICAL EXAM']
initialassessment = ['Initial Assessment', 'INITIAL ASSESSMENT']
edcourse = ['ED Course', 'ED course', 'ED COURSE']

df['history_text'] = df['note_text_processed'].apply(lambda x: extract_text(x, chiefcomplaint, physicalexam)) 

df['examination_text'] = df['note_text_processed'].apply(lambda x: extract_text(x, physicalexam, initialassessment) 
                                               if any(s in x for s in initialassessment) else extract_text(x, physicalexam, edcourse) 
                                               if any(s in x for s in edcourse) else 'unable_to_extract')

df['assessment_plan_text'] = df['note_text_processed'].apply(lambda x: extract_initialassessment_to_end(x, initialassessment, edcourse))

##Get token count for each text column
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens

df['note_text_processed_tokens'] = df['note_text_processed'].apply(lambda x: num_tokens_from_string(x))
print('Next')
df['history_text_tokens'] = df['history_text'].apply(lambda x: num_tokens_from_string(x))


##Save processed dataset
df.to_csv('path/ed_notes_edprovider_adults_master_processed.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')

##Create a modified version of the master df which contains nulls etc excluded:

#Remove acuitylevel == '*Unspecified'
print(df.shape)
df_refined = df[df['acuitylevel'] != '*Unspecified']
print(df_refined.shape)

#Remove null values ['history_text', 'examination_text', 'assessment_plan_text']
print(df_refined.shape)
for text in ['history_text', 'examination_text', 'assessment_plan_text']:
    print('Removing nulls from', text)
    df_refined = df_refined[df_refined[text].notnull()]
    print(df_refined.shape)

#Similarly, remove 'unable_to_extract' from ['history_text', 'examination_text', 'assessment_plan_text']
for text in ['history_text', 'examination_text', 'assessment_plan_text']:
    print('Removing unable_to_extract from', text)
    df_refined = df_refined[df_refined[text] != 'unable_to_extract']
    print(df_refined.shape)
    
#For completeness, remove '' from ['history_text', 'examination_text', 'assessment_plan_text']
for text in ['history_text', 'examination_text', 'assessment_plan_text']:
    print('Removing unable_to_extract from', text)
    df_refined = df_refined[df_refined[text] != '']
    print(df_refined.shape)

#Include only rows with history_text_tokens of 2000 or fewer
df_refined = df_refined[df_refined['history_text_tokens'] <= 2000]


##Create balanced dataset from df_refined
#Select only desired columns
df_testB = df_refined[['visit_occurrence_id', 'acuitylevel', 'history_text']]

def sample_acuity_scores_B(df_master):
    #Sample 1000 pairs for each of the 10 categories
    #Have to sample with replacement as we only have 1200 Immediate and 4537 Non-Urgent patients
    df_master = df_master[['visit_occurrence_id', 'acuitylevel', 'history_text']]

    df_master_A = df_master.copy()
    df_master_B = df_master.copy()

    df_master_A = df_master_A.rename(columns = {'visit_occurrence_id':'visit_occurrence_id_A', 'acuitylevel':'acuitylevel_A', 'history_text':'history_text_A'})
    df_master_B = df_master_B.rename(columns = {'visit_occurrence_id':'visit_occurrence_id_B', 'acuitylevel':'acuitylevel_B', 'history_text':'history_text_B'})
    
    #Create blank sample dataframes which we append to after each iteration of acuitylevel and then concat on axis = 1 to generate paired dataset
    df_sample_A  = pd.DataFrame(columns = ['visit_occurrence_id_A', 'acuitylevel_A', 'history_text_A'])
    df_sample_B  = pd.DataFrame(columns = ['visit_occurrence_id_B', 'acuitylevel_B', 'history_text_B'])

    sample_A_acuity_list = ['Immediate', 'Immediate', 'Immediate', 'Immediate', 'Emergent', 'Emergent', 'Emergent', 'Urgent', 'Urgent', 'Less Urgent']
    sample_B_acuity_list = ['Emergent', 'Urgent', 'Less Urgent', 'Non-Urgent', 'Urgent', 'Less Urgent', 'Non-Urgent', 'Less Urgent', 'Non-Urgent', 'Non-Urgent']

    for acuity in sample_A_acuity_list:
        #Sample with replacement during initial sampling as well as globally (due to nature of acuitylevel selection then sampling)
        #Create new random_seed each time (so can save seed for replication purposes, and get different random samples each time)
        seed_value = random.randint(0, 10000)
        A = df_master_A[df_master_A['acuitylevel_A'] == acuity].sample(1000, replace = True, random_state = seed_value).reset_index(drop = True)
        A['seed_A'] = seed_value
        df_sample_A = pd.concat([df_sample_A.reset_index(drop = True), A])
    for acuity in sample_B_acuity_list:
        #Sample with replacement during initial sampling as well as globally (due to nature of acuitylevel selection then sampling)
        seed_value = random.randint(0, 10000)       
        B = df_master_B[df_master_B['acuitylevel_B'] == acuity].sample(1000, replace = True, random_state = seed_value).reset_index(drop = True)
        B['seed_B'] = seed_value
        df_sample_B = pd.concat([df_sample_B.reset_index(drop = True), B])
    #Finally, concat columns of df_sample_A and df_sample_B
    df_output = pd.concat([df_sample_A.reset_index(drop = True), df_sample_B.reset_index(drop = True)], axis = 1)
    return df_output

df_testB_output = sample_acuity_scores_B(df_testB)


##Randomly shuffle the _A and _B group of columns so that A is not always before B (and re-label _A and _B after)
#Essentially, don't want the higher acuity patients to always be in A as this may bias the ChatGPT API

#Sample split the dataset in half, reverse the column names, then concat
#Retrieve the combined visit_occurrence_id _A and _B
df_testB_output['visit_occurrence_id_combined'] = df_testB_output['visit_occurrence_id_A'] + '_' +  df_testB_output['visit_occurrence_id_B']

df_testB_output_halfA = df_testB_output.sample(5000, random_state = 7, replace = False)
df_testB_output_halfB = df_testB_output[~df_testB_output['visit_occurrence_id_combined'].isin(df_testB_output_halfA['visit_occurrence_id_combined'].tolist())]

#Rename df_testB_output_halfB columns
df_testB_output_halfB = df_testB_output_halfB.rename(columns = {'visit_occurrence_id_A':'visit_occurrence_id_B', 'visit_occurrence_id_B':'visit_occurrence_id_A',
                                                      'acuitylevel_A':'acuitylevel_B', 'acuitylevel_B':'acuitylevel_A',
                                                      'history_text_A':'history_text_B', 'history_text_B':'history_text_A',
                                                      'seed_A':'seed_B', 'seed_B':'seed_A'})

#Concatenate everything back together again
df_testB_output = pd.concat([df_testB_output_halfA, df_testB_output_halfB])

#Confirm visit_occurrence_id_B and visit_occurrence_id_A do not match
print((df_testB_output['visit_occurrence_id_A'] == df_testB_output['visit_occurrence_id_B']).sum())

#Check for no duplicated visit_occurrence_id_combined
print('\n Number of duplicated visit_occurrence_id_combined:', (df_testB_output['visit_occurrence_id_combined'].duplicated().sum()), '\n')

#Confirm acuitylevel_B and acuitylevel_A do not match
print((df_testB_output['acuitylevel_A'] == df_testB_output['acuitylevel_B']).sum())

##Count the nunique matched pairs
df_testB_output_acuity_counts = df_testB_output.groupby(['acuitylevel_A', 'acuitylevel_B']).size().reset_index(name = 'count').sort_values(by = ['count'], ascending = False).reset_index(drop = True)
# Create a new column that combines the labels in alphabetical order
df_testB_output_acuity_counts['labels'] = df_testB_output_acuity_counts[['acuitylevel_A', 'acuitylevel_B']].apply(sorted, axis=1).apply('-'.join)
# Group by the new column and sum the counts
df_testB_output_acuity_counts = df_testB_output_acuity_counts.groupby('labels').agg({'count': 'sum'}).sort_values(by = ['count']).reset_index()
df_testB_output_acuity_counts
#10 groups of 1000 pairs as desired

##Map numeric ESI score to allow easier interpretation of highest acuity
ESI_score_dict = {'Immediate':1, 'Emergent':2, 'Urgent':3, 'Less Urgent':4, 'Non-Urgent':5}
df_testB_output['acuitylevel_int_A'] = df_testB_output['acuitylevel_A'].map(ESI_score_dict)
df_testB_output['acuitylevel_int_B'] = df_testB_output['acuitylevel_B'].map(ESI_score_dict)

print(df_testB_output.acuitylevel_int_A.value_counts())
print(df_testB_output.acuitylevel_int_B.value_counts())

##Create prompt
initial_prompt = "You are an Emergency Department physician. Below are the symptoms of two different patients presenting to the Emergency Department, Patient A and Patient B. Please return which patient is of the highest acuity between these two patients. Please return one of two answers: '0: Patient A is of higher acuity' '1: Patient B is of higher acuity' Please do not return any additional explanation."
df_testB_output['prompt'] = initial_prompt + '  \n  Patient A: """' + df_testB_output['history_text_A'] + '"""  \n  Patient B: """' + df_testB_output['history_text_B'] + '"""'

##Confirm no null prompts
print(df_testB_output[df_testB_output['prompt'].isnull()].shape, '- should equal 0')

#Count number of tokens in intro section of prompt
intro = "You are an Emergency Department physician. Below are the symptoms of two different patients presenting to the Emergency Department, Patient A and Patient B. Please return which patient is of the highest acuity between these two patients. Please return one of two answers: '0: Patient A is of higher acuity' '1: Patient B is of higher acuity' Please do not return any additional explanation.   \n  Patient A: \"\"\" \"\"\"   \n  Patient B: \"\"\" \"\"\" "

print(num_tokens_from_string(intro))
#94

#Add this to the (minimum) 11 tokens from the (desired) output = 105
#Hence absolute maximum prompt token length = 4097 (max tokens in model) - 105 = 3992

#Count number of tokens overall in full (combined) prompt
df_testB_output['prompt_tokens'] = df_testB_output['prompt'].apply(lambda x: num_tokens_from_string(x))

#Test if max count of prompt_tokens is > 3992
print(max(df_testB_output['prompt_tokens']))
#3775 - hence, fine

#Calculate difference between acuitylevel_int_B and acuitylevel_int_A
#Positive diff = B is less acute than A
df_testB_output['acuity_level_int_BminusA'] = df_testB_output['acuitylevel_int_B'] - df_testB_output['acuitylevel_int_A']
#Examine
print(df_testB_output['acuity_level_int_BminusA'].value_counts())

#Label the higher_acuity_patient
df_testB_output.loc[df_testB_output['acuity_level_int_BminusA'] > 0, 'higher_acuity_patient'] = 'A'
df_testB_output.loc[df_testB_output['acuity_level_int_BminusA'] < 0, 'higher_acuity_patient'] = 'B'

print(df_testB_output['higher_acuity_patient'].value_counts())
#A    5000
#B    5000

#Shuffle df (to get better representation of samples with subset)
df_testB_output = df_testB_output.sample(frac=1, random_state=7)

#Save sample master_df
df_testB_output.to_csv('path/ed_notes_edprovider_adults_master_processed_df_testB_output.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')

##Onto RAE_chatgpt_ed_acuity_adults_openai_final_sanitised_gpt4.py