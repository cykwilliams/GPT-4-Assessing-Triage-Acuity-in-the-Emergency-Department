%%configure -f
 
{
    "conf":{"spark.driver.memory":"25g",
        "spark.driver.maxResultSize":"25g",
        "spark.executor.memory":"25g",
        "spark.executor.cores":"6",
        "spark.cores.max": "12"
    }
}

##CYKW functions
#pip install pyspark_dist_explore

#import packages
from pyspark.sql.functions import col
from pyspark.sql.functions import lower
from pyspark.sql.functions import isnan
from pyspark.sql.functions import lit, datediff, to_date
from pyspark.sql.functions import when, count
from pyspark.sql.functions import year, to_timestamp
#from math import log
from pyspark.sql.functions import log
from pyspark.sql.types import IntegerType


import pyspark.sql.functions as F

import pandas as pd
import numpy as np

def print_if(*args):
  if display_print == 'Y':
    print(*args)
  else:
    print('CYKW: Output not printed; to print these outputs, change display_print = \'Y\'')
    pass

def shape(spark_df):
  if display_print == 'Y':
    print_if('(', spark_df.count(), ',', len(spark_df.columns), ')')
  else:
    pass
  
def value_counts(spark_df, column):
  if display_print == 'Y':
    print_if(spark_df.groupBy(column).count().orderBy('count', ascending = False).show(200, truncate = False))
    print_if('(Returns the first 200 results)')
  else:
    pass

def value_counts_return_df(spark_df, column):
  return spark_df.groupBy(column).count().orderBy('count', ascending = False)
  
  
def merge(spark_df1, spark_df2, left_on, right_on, how):
  #To prevent duplication 'on' columns, need to rename and then drop:
  spark_df2 = spark_df2.withColumnRenamed(right_on, 'right_on')
  merged = spark_df1.join(spark_df2, spark_df1[left_on] == spark_df2['right_on'], how).drop('right_on')
  return merged
  
def head(spark_df, n):
  if display_print == 'Y':
    print_if(spark_df.show(n, vertical = True, truncate = False))
  else:
    pass
  
def sample(spark_df, n):
  if display_print == 'Y':
    print_if(spark_df.sample(fraction=1.0).show(n, vertical = True, truncate = False))
  else:
    pass
  
def sample_df(spark_df, n):
    n_denominator = spark_df.count()
    return spark_df.sample(False, fraction=(n/denominator), seed = 42).limit(n)

  
def nunique(spark_df, column):
  print_if(spark_df.select(column).distinct().count())
  
def sort_values(spark_df, column1, column2, how):
  if how == 'descending':
    if column2 is None:
      sorted = spark_df.sort(col(column1).desc())
    if column2 is not None:
      sorted = spark_df.sort(col(column1).desc(), col(column2).desc())
  if how == 'ascending':
    if column2 is None:
      sorted = spark_df.sort(col(column1).asc())
    if column2 is not None:
      sorted = spark_df.sort(col(column1).asc(), col(column2).asc())
  return sorted

def drop_duplicates(spark_df, subset):
  print_if('drop_duplicates: keep = \'first\' ')
  dropped = spark_df.drop_duplicates(subset)
  return dropped

def count_null(spark_df, column):
  if display_print == 'Y':
    print_if(spark_df.filter((spark_df[column] == "")|spark_df[column].isNull()).count())
  else:
    pass
  
def count_notnull(spark_df, column):
  if display_print == 'Y':
    print_if(spark_df.filter(col(column).isNotNull()).count())
  else:
    pass
  
def return_notnull_df(spark_df, column):
  return spark_df.filter(col(column).isNotNull())

def return_isnull_df(spark_df, column):
  return spark_df.filter(col(column).isNull())
  
def register_table(spark_df, name):
  spark_df.registerTempTable(name)

def rename_columns(df, columns):
    if isinstance(columns, dict):
        return df.select(*[F.col(col_name).alias(columns.get(col_name, col_name)) for col_name in df.columns])
    else:
        raise ValueError("'columns' should be a dict, like {'old_name_1':'new_name_1', 'old_name_2':'new_name_2'}")
        
def isnull(df, columns):
  if display_print == 'Y':
    print_if(df.filter(col(columns).isNull()).count())
  else:
    pass
  
def test_column_equality(df, column_1, column_2):
  df_test_column_equality = df.withColumn("match_y_n", col(column_1) == col(column_2)).select(col("match_y_n"))
  value_counts(df_test_column_equality, 'match_y_n')
  
def get_timestamp_year(df, timestamp_column, new_column, timestamp_format):
  df = df.withColumn(new_column,year(to_timestamp(timestamp_column, timestamp_format)))
  return df

def label_care_site_id(df):
  care_site = spark.sql("SELECT * FROM omop_deid.care_site")
  location = spark.sql("SELECT * FROM omop_deid.location")

  care_site_full = merge(care_site, location, left_on = 'location_id', right_on = 'location_id', how = 'outer')
  df = merge(df, care_site_full.select(col('care_site_id'), col('location_source_value')), left_on = 'care_site_id', right_on = 'care_site_id', how = 'left')
  return df

def hist(df, column, scale):
  #Copied from here https://stackoverflow.com/questions/39154325/pyspark-show-histogram-of-a-data-frame-column
  from pyspark_dist_explore import hist
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots()
  
  if scale == 'log':
    print('log scale y-axis')
    return hist(ax, df.select((log(column))), bins = 20, color=['blue'])
  else: 
    return hist(ax, df.select((col(column).cast(IntegerType()))), bins = 20, color=['blue'])
    #return hist(ax, df.select((col(column))), bins = 20, color=['blue'])

def calc_df_stats(df):
  if display_print == 'Y':
    print_if('Number of rows:', df.count())
    print_if('Number of unique patients:')
    nunique(df, 'person_id')
    print_if('Number of unique visit_occurrences:')
    nunique(df, 'visit_occurrence_id')
                
##Set whether to print values (takes up time)
display_print = 'Y'

##Display non-truncated text
spark.conf.set("spark.sql.repl.eagerEval.maxToStringFields", 1000)


import pandas as pd
pd.set_option('display.max_rows', 200)

path = 'path'

ed_table = spark.read.options(header='true', inferschema='true').parquet(path + 'ed_table/')
print(ed_table.count())
print(nunique(ed_table, 'visit_occurrence_id'))

##Select only relevant columns:
columns_to_select = ['person_id', 'ed_visit_occurrence_id', 'person_id2', 'agekey', 'agekeyvalue', 'admission_visit_occurrence_id', 'visit_occurrence_id', 'arrival_datetime', 'admissiondecision_datetime', 'departure_datetime', 'disposition_datetime', 'departmentname', 'departmentspecialty', 'admissiondepartmentname', 'admissiondepartmentspecialty', 'eddisposition', 'dischargedisposition', 'acuitylevel']
ed_table = ed_table.select([col(column) for column in columns_to_select])
print(ed_table.columns)

patient_key_table = spark.read.options(header='true', inferschema='true').parquet(path + 'patient_key_table/')
print(patient_key_table.columns)

#Add in sex/race/ethnicity
ed_table = merge(ed_table, patient_key_table.select(col('person_id'), col('sex'), col('ethnicity'), col('firstrace'), col('multiracial')), left_on = 'person_id', right_on = 'person_id', how = 'left')

#Merge agekey with keys in duration_key_table to retrieve age
duration_key_table = spark.read.options(header='true', inferschema='true').parquet(path + 'duration_key_table/')

ed_table = merge(ed_table, duration_key_table.select(col('durationkey'), col('days')), left_on = 'agekey', right_on = 'durationkey', how = 'left')
ed_table = ed_table.withColumn('edvisit_age', col('days') / 365.25)

note_metadata = spark.read.options(header='true', inferschema='true').parquet(path + 'note_metadata')
print(note_metadata.count())

note_text = spark.read.options(header='true', inferschema='true').parquet(path + 'note_text')
print(note_text.count())

#Select relevant columns
notes = note_metadata.select(col('person_id'), col('deid_note_key'), col('visit_occurrence_id'), col('note_type'), col('encounter_type'), col('enc_dept_name'), col('enc_dept_specialty'),col('auth_prov_type'), col('prov_specialty'), col('deid_service_date'))
#Add note_text
notes = merge(notes, note_text, left_on = 'deid_note_key', right_on = 'deid_note_key', how = 'left')

notes = rename_columns(notes, {'person_id':'person_id_notes'})

#Further reduce columns in ed_table
print(ed_table.columns)
columns_to_select2 = ['person_id', 'ed_visit_occurrence_id', 'person_id2', 'admission_visit_occurrence_id', 'visit_occurrence_id', 'arrival_datetime', 'admissiondecision_datetime', 'departure_datetime', 'disposition_datetime', 'departmentname', 'departmentspecialty', 'admissiondepartmentname', 'admissiondepartmentspecialty', 'eddisposition', 'acuitylevel', 'sex', 'ethnicity', 'firstrace', 'multiracial', 'days', 'edvisit_age']
ed_table_selected = ed_table.select([col(column) for column in columns_to_select2])
ed_notes = merge(ed_table_selected, notes, left_on = 'visit_occurrence_id', right_on = 'visit_occurrence_id', how = 'inner')

#Examine difference in number of unique encounters with notes available compared to ed_table
nunique(ed_table, 'visit_occurrence_id')
nunique(ed_notes, 'visit_occurrence_id')
print(ed_notes.count())

nunique(return_isnull_df(ed_notes, 'edvisit_age'), 'visit_occurrence_id')
ed_notes_adults = ed_notes.filter(col('edvisit_age') >= 18)

nunique(ed_notes_adults, 'visit_occurrence_id')
print(ed_notes_adults.count())

ed_notes_adults_edprovider = ed_notes_adults.filter(col('note_type') == 'ED Provider Notes')

nunique(ed_notes_adults_edprovider, 'visit_occurrence_id')
print(ed_notes_adults_edprovider.count())
value_counts(ed_notes_edprovider, 'prov_specialty')

#Filter by prov_specialty (including only Emergency Medicine and UCSF)
ed_notes_edprovider = ed_notes_edprovider.filter((col('prov_specialty') == 'Emergency Medicine')|(col('prov_specialty') == 'UCSF'))

nunique(ed_notes_edprovider_adults_filtered, 'visit_occurrence_id')
print(ed_notes_edprovider_adults_filtered.count())

ed_notes_edprovider_adults.write.mode("overwrite").option("mergeSchema", "true").parquet("path/ed_notes_edprovider_adults_master.parquet")

##Onto chatgpt_ed_acuity_adults_preprocessing_final_sanitised.py

