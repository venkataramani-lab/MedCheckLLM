# -*- coding: utf-8 -*-


#%%
import pandas as pd
import os
import openai
import tiktoken
import json
import requests
from bs4 import BeautifulSoup
import random


#%%

enc = tiktoken.encoding_for_model("gpt-4")

with open("D:\\Data\\Marc(D)\\k") as k:
    content = k.read()
os.environ["OPENAI_API_KEY"] = content
openai.api_key = os.getenv("OPENAI_API_KEY")


#%% Functions

def getExampleMedicalReport(diagnosis, model="gpt-4-0613"):
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[
        {"role": "user", "content": "Please write an example medical report for a person with the diagnosis " + diagnosis + ".\n" }
      ])
    ret = completion["choices"][0]["message"]["content"]
    return ret


def selectGuideline(report, model="gpt-4-0613"):
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[
        {"role": "user", "content": '''Please examine the following medical report and provide:
- The leading symptom
- potential syndrome 
- The diagnosis
- The name of a relevant, established medical guideline for the leading symptom.

\n Do not return actual medical advice but the name of a relevant medical guideline.

\nMedical Report:\n''' + report + '''\n\n'''}

      ], 
    functions = [
        {
            "name": "give_leadingsymdiagguideline",
            "description": """

          - The leading symptom
          - The diagnosis
          - A relevant medical guideline for further evaluation
            """,
            "parameters": {"type":"object", 
                           "properties":{
                                          "leading_symptom": {"type":"string"},
                                          "diagnosis": {"type": "string"},
                                          "guideline":{"type":"string"}}
                                                                           }

            }
        ], 
    function_call={"name":"give_leadingsymdiagguideline"}
    )
    return json.loads(completion["choices"][0]["message"]["function_call"]["arguments"])

# Get Diagnostic Criteria
def get_diagnosticCriteria(ichd_library, key):
    href = ichd_library[key]        
    resp = requests.get(href) 
    resp = BeautifulSoup(resp.text, "html.parser")
    criterias = resp.find('ol', {'style': 'list-style-type: upper-alpha;'})
    #items = criterias.find_all('li', recursive=False)
    #letters = "ABCDEFGHIJKLMNOP"
    #for i, it in enumerate(items):
    #    print(letters[i],i,  it.text)   
    text = criterias.text
    return text
    
# Classification Library
def accessGuidelines(link = "https://ichd-3.org/classification-outline/"):
    html = requests.get(link)

    soup = BeautifulSoup(html.text, "html.parser")

    a_tags = soup.find_all('a')

    ichd_library = {}

    for a in a_tags:  
        if len(a.text) <4:
            continue
        ax = a.text[:7]
        first = ax[:ax.find(".")]
        rest = ax[ax.find(".")+1:]
        if rest.find(".")>-1:
             continue      
        second = rest[:rest.find(" ")]    
        if first.isdigit() and second.isdigit():
            href = a.get("href")
            print("Target: ",a.text )
            ichd_library.update({a.text: href})
    
    access_keys= []
    for i, key in enumerate(ichd_library.keys()):
        print(key)
        try:
            crits = get_diagnosticCriteria(ichd_library, key)
            access_keys.append(key)
        except Exception as e: 
            print(str(e))
    
    ichd_library = {key: ichd_library[key] for key in access_keys if key in ichd_library}

    return ichd_library

def selectKey(KEYS,report, model="gpt-4-0613"):
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[
        {"role": "user", "content": "You are supposed to extract a diagnosis of a medical report. Of the choices, return the approriate headache type key for the diagnosis in the following medical report:\n\nKeys:" + KEYS +"\nMedical Report:\n" + report + "\n\n Please return ONLY the key, so that I can access the dictionary directly., e.g.:\n14.1 Headache not elsewhere classified"}

      ])
    ret = completion["choices"][0]["message"]["content"]
    return ret


def turnIntoChecklist(guideline_out, model="gpt-4-0613"):
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[
        {"role": "user", "content": f'''
         I will provide you with part of a medical guideline. If it is in a format that could be used as a checklist, return the word 'CHECKLIST'. Only return the word 'CHECKLIST', not the actual checklist itself.
         If it is a continuous, long text with complete sentences, try to extract a checklist from the guideline and return your checklist.
         \n
         Guideline:\n
             {guideline_out} 
         '''}

      ])
    ret = completion["choices"][0]["message"]["content"]
    return ret


def evaluateLetter(letter, checklist,diagnosis, model="gpt-4-0613"):
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[
        {"role": "user", "content": """
           I will provide you with a checklist guideline and a medical report. 
           Please assess the doctor's letter based on the checklist and consider the following: 
               Checklist items are usually numbered  A,B,C,D,E ... 
               Understand what is meant by for example: Any headache fulfilling criterion B and C.
               Then assess the letter and  
               
               return your results: 
             
           - For each checklist item separatedly: 
               - Whether the checklist item was thoroughly addressed in the doctor's letter (0-5: not covered at all (0) - comprehensively covered (5)).
               - Comment.
           - Additional comments.

         
         """}, 
        {"role": "user", "content": "Checklist for "+diagnosis+":\n" + checklist + "\n---\nLetter:\n" + letter}, 
      ], 
    functions = [
        {
            "name": "give_output_of_evaluation",
            "description": """

            - Comprehensiveness of coverage of checklist item: (0-5)
            - Comments on checklist item (string)
            - Additional comments
            """,
            "parameters": {"type":"object", 
                           "properties":{
                               "checklist_completion_status": {

                               "type":"array",
                               "items": {
                                   "type":"object", 
                                   "properties":{
                                          "item": {"type":"string"},
                                          "comprehensiveness": {"type": "integer"},
                                          "comment":{"type":"string"}}
                                                                           }
                                                                 }
                                   
                               , 

                           "further_comments": {"type": "string"}
                           
                           
                               }# properties  
                           
                           
                                   }
            }
                               
        ], 
    function_call={"name":"give_output_of_evaluation"}
    )
    return json.loads(completion["choices"][0]["message"]["function_call"]["arguments"])



def correctDiagVsNot(letter, model="gpt-4-0613"):
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[
        {"role": "user", "content": """
           I will provide you with a medical report. 
           Please assess whether the doctor's letter identified the correct diagnosis.
            Return the following results:
               - The diagnosis stated in the doctor's letter
               - The diagnosis that you believe the patient actually has
               - whether the stated and actual diagnosis are the same (yes/no)
               - Additional comments.
         
         """}, 
        {"role": "user", "content": "Letter:\n" + letter}, 
      ], 
    functions = [
        {
            "name": "give_output_of_diagnosis_evaluation",
            "description": """
             - The diagnosis stated in the doctor's letter
             - The diagnosis that you believe the patient actually has
             - whether the stated and actual diagnosis are the same (yes/no)
            - Additional comments.
            """,
            "parameters": {"type":"object", 
                           "properties":{
                               "diagnosis_accuracy": {
                                   "type":"object", 
                                   "properties":{
                                          "stated_diagnosis": {"type":"string"},
                                          "actual_diagnosis": {"type": "string"},
                                           "diagnosis_correctness":{"type":"string"},
                                          "comment":{"type":"string"}}
                                                                           }
                                                                 
                }# properties  
                                   }
            }
                               
        ], 
    function_call={"name":"give_output_of_diagnosis_evaluation"}
    )
    return json.loads(completion["choices"][0]["message"]["function_call"]["arguments"])



    
def analyzeLetter(letter, letter_key): 
    KEYS= "'1.1 Migraine without aura', '1.3 Chronic migraine', '1.5 Probable migraine', '2.1 Infrequent episodic tension-type headache', '3.1 Cluster headache', '3.2 Paroxysmal hemicrania', '3.3 Short-lasting unilateral neuralgiform headache attacks', '3.4 Hemicrania continua', '3.5 Probable trigeminal autonomic cephalalgia', '4.1 Primary cough headache', '4.2 Primary exercise headache', '4.3 Primary headache associated with sexual activity', '4.4 Primary thunderclap headache', '4.7 Primary stabbing headache', '4.8 Nummular headache', '4.9 Hypnic headache', '4.10 New daily persistent headache (NDPH)', '5.1 Acute headache attributed to traumatic injury to the head', '5.2 Persistent headache attributed to traumatic injury to the head', '5.3 Acute headache attributed to whiplash1', '5.4 Persistent headache attributed to whiplash', '5.5 Acute headache attributed to craniotomy', '5.6 Persistent headache attributed to craniotomy', '6.4 Headache attributed to arteritis', '6.5 Headache attributed to cervical carotid or vertebral artery disorder', '6.9 Headache attributed to pituitary apoplexy', '7.1 Headache attributed to increased cerebrospinal fluid (CSF) pressure', '7.2  Headache attributed to low cerebrospinal fluid (CSF) pressure', '7.3 Headache attributed to non-infectious inflammatory intracranial disease', '7.4 Headache attributed to intracranial neoplasia', '7.5 Headache attributed to intrathecal injection', '7.6 Headache attributed to epileptic seizure', '7.7 Headache attributed to Chiari malformation type I (CM1)', '7.8 Headache attributed to other non-vascular intracranial disorder', '8.2 Medication-overuse headache (MOH)', '10.1 Headache attributed to hypoxia and/or hypercapnia', '10.2 Dialysis headache', '10.4 Headache attributed to hypothyroidism', '10.5 Headache attributed to fasting', '10.6 Cardiac cephalalgia', '10.7 Headache attributed to other disorder of homoeostasis', '11.1 Headache attributed to disorder of cranial bone', '11.4 Headache attributed to disorder of the ears', '11.6 Headache attributed to disorder of the teeth', '11.7 Headache attributed to temporomandibular disorder (TMD)', '11.8 Headache or facial pain attributed to inflammation of the stylohyoid ligament', '11.9 Headache or facial pain attributed to other disorder of cranium, neck, eyes, ears, nose, sinuses, teeth, mouth or other facial or cervical structure', '12.1 Headache attributed to somatization disorder1', '12.2 Headache attributed to psychotic disorder', '13.1 Pain attributed to a lesion or disease of the trigeminal nerve', '13.4 Occipital neuralgia', '13.5 Neck-tongue syndrome', '13.6 Painful optic neuritis', '13.7 Headache attributed to ischaemic ocular motor nerve palsy', '13.8 Tolosa-Hunt syndrome', '13.9 Paratrigeminal oculosympathetic (Raeder’s) syndrome', '13.10 Recurrent painful ophthalmoplegic neuropathy', '13.11 Burning mouth syndrome (BMS)', '13.12 Persistent idiopathic facial pain (PIFP)', '14.1 Headache not elsewhere classified', '14.2 Headache unspecified'"
    
    print("Identifying diagnosis..")
    key = selectKey(KEYS, report = letter)
    print("Identified diagnosis:", key, ". Now getting diagnostic criteria...")
    
    criteria = get_diagnosticCriteria(ichd_library,  key)
    print("Criteria extracted. Now evaluating..")
    evals = evaluateLetter(letter, checklist=criteria,  diagnosis=key)
    print("Evaluation done. Now saving..")
    evaluations = evals["checklist_completion_status"]
    df_list =[]
    for checklist_item in evaluations:
        item = checklist_item["item"]
        comp = checklist_item["comprehensiveness"]
        comm = checklist_item["comment"]
        print(item, "\n", comp, "\n",comm)
        row_dict = {"letter_key": letter_key,
            "identified_diagnosis":key, 
                    "criteria":criteria,
                       "letter": letter, 
                       "evals": json.dumps(evals), 
                       "item": item,
                       "comprehensiveness": comp,
                       "comment": comm
                       }
        df = pd.DataFrame([row_dict])
        df_list.append(df)
    bound = pd.concat(df_list)
    bound.to_csv("analyzed_v2_" + key + ".csv")
    return bound





#%% Run

ichd_library = accessGuidelines()


samples = random.sample(list(ichd_library.keys()),20)
samples  =  ['7.5 Headache attributed to intrathecal injection',
 '3.1 Cluster headache',
 '4.9 Hypnic headache',
 '11.6 Headache attributed to disorder of the teeth',
 '2.1 Infrequent episodic tension-type headache',
 '6.9 Headache attributed to pituitary apoplexy',
 '12.2 Headache attributed to psychotic disorder',
 '13.11 Burning mouth syndrome (BMS)',
 '1.5 Probable migraine',
 '7.1 Headache attributed to increased cerebrospinal fluid (CSF) pressure',
 '7.3 Headache attributed to non-infectious inflammatory intracranial disease',
 '13.4 Occipital neuralgia',
 '11.9 Headache or facial pain attributed to other disorder of cranium, neck, eyes, ears, nose, sinuses, teeth, mouth or other facial or cervical structure',
 '7.4 Headache attributed to intracranial neoplasia',
 '5.6 Persistent headache attributed to craniotomy',
 '4.2 Primary exercise headache',
 '13.7 Headache attributed to ischaemic ocular motor nerve palsy',
 '5.4 Persistent headache attributed to whiplash',
 '7.8 Headache attributed to other non-vascular intracranial disorder',
 '3.5 Probable trigeminal autonomic cephalalgia']


dictionary  = {}
for sample in samples:
    print(sample)
    diagnosis = sample
    diagnosis = diagnosis[diagnosis.find(" ")+1:]
    mr = getExampleMedicalReport(diagnosis= diagnosis)
    dictionary.update({diagnosis:mr})  


with open("sample_mrs_dictionary.json", "w") as file:
    json.dump(dictionary, file)

eval_lists = []
for letter_key in dictionary:
    print(letter_key)
    try: 
        df = analyzeLetter(letter=dictionary[letter_key], letter_key = letter_key)
        eval_lists.append(df)
    except Exception as e:
        print(str(e))
    

        

all_evals = pd.concat(eval_lists)

#all_evals.to_csv("all_evals_v3.csv")

#%%
# create letters with wrong diagnosis
os.chdir("Y:\\NGS\\26.Projects\\Checklists\\V3")

all_evals = pd.read_csv("all_evals_v3.csv")
df = all_evals[all_evals["letter"].duplicated() == False]
df
df.to_csv("letters_with_wrong_diagnosis.csv")


#%%

#KEYS = "'1.1 Migraine without aura', '1.3 Chronic migraine', '1.5 Probable migraine', '2.1 Infrequent episodic tension-type headache', '3.1 Cluster headache', '3.2 Paroxysmal hemicrania', '3.3 Short-lasting unilateral neuralgiform headache attacks', '3.4 Hemicrania continua', '3.5 Probable trigeminal autonomic cephalalgia', '4.1 Primary cough headache', '4.2 Primary exercise headache', '4.3 Primary headache associated with sexual activity', '4.4 Primary thunderclap headache', '4.7 Primary stabbing headache', '4.8 Nummular headache', '4.9 Hypnic headache', '4.10 New daily persistent headache (NDPH)', '5.1 Acute headache attributed to traumatic injury to the head', '5.2 Persistent headache attributed to traumatic injury to the head', '5.3 Acute headache attributed to whiplash1', '5.4 Persistent headache attributed to whiplash', '5.5 Acute headache attributed to craniotomy', '5.6 Persistent headache attributed to craniotomy', '6.4 Headache attributed to arteritis', '6.5 Headache attributed to cervical carotid or vertebral artery disorder', '6.9 Headache attributed to pituitary apoplexy', '7.1 Headache attributed to increased cerebrospinal fluid (CSF) pressure', '7.2  Headache attributed to low cerebrospinal fluid (CSF) pressure', '7.3 Headache attributed to non-infectious inflammatory intracranial disease', '7.4 Headache attributed to intracranial neoplasia', '7.5 Headache attributed to intrathecal injection', '7.6 Headache attributed to epileptic seizure', '7.7 Headache attributed to Chiari malformation type I (CM1)', '7.8 Headache attributed to other non-vascular intracranial disorder', '8.2 Medication-overuse headache (MOH)', '10.1 Headache attributed to hypoxia and/or hypercapnia', '10.2 Dialysis headache', '10.4 Headache attributed to hypothyroidism', '10.5 Headache attributed to fasting', '10.6 Cardiac cephalalgia', '10.7 Headache attributed to other disorder of homoeostasis', '11.1 Headache attributed to disorder of cranial bone', '11.4 Headache attributed to disorder of the ears', '11.6 Headache attributed to disorder of the teeth', '11.7 Headache attributed to temporomandibular disorder (TMD)', '11.8 Headache or facial pain attributed to inflammation of the stylohyoid ligament', '11.9 Headache or facial pain attributed to other disorder of cranium, neck, eyes, ears, nose, sinuses, teeth, mouth or other facial or cervical structure', '12.1 Headache attributed to somatization disorder1', '12.2 Headache attributed to psychotic disorder', '13.1 Pain attributed to a lesion or disease of the trigeminal nerve', '13.4 Occipital neuralgia', '13.5 Neck-tongue syndrome', '13.6 Painful optic neuritis', '13.7 Headache attributed to ischaemic ocular motor nerve palsy', '13.8 Tolosa-Hunt syndrome', '13.9 Paratrigeminal oculosympathetic (Raeder’s) syndrome', '13.10 Recurrent painful ophthalmoplegic neuropathy', '13.11 Burning mouth syndrome (BMS)', '13.12 Persistent idiopathic facial pain (PIFP)', '14.1 Headache not elsewhere classified', '14.2 Headache unspecified'"
#len(KEYS.split("',"))
#Out[4]: 61

os.chdir("Y:\\NGS\\26.Projects\\Checklists\\V3")
os.listdir()
data = pd.read_excel("letters_with_wrong_diagnosis.xlsx")

evals_list =[]
for i , row in data.iterrows():
    print(i)
    print(row["letter_key"])
    key = row["letter_key"]
    letter = row["letter"]
    try:
        out = correctDiagVsNot(letter, model="gpt-4-0613")
    except: 
        continue
    out = out["diagnosis_accuracy"]
    
    row_dict = {
        "original_key":key,
        "letter": letter,
        "stated_diagnosis":out["stated_diagnosis"],
        "actual_diagnosis":out["actual_diagnosis"],
        "diagonsis_correctness":out["diagnosis_correctness"],
        "comment": out["comment"]
        }

    df = pd.DataFrame([row_dict])
    evals_list.append(df)


correct_letters=pd.concat(evals_list)
#correct_letters.to_csv("correct_letters_diagnosis_correctness.csv")

#%%
# extraction of checklist items 
os.getcwd()
os.listdir()
data = pd.read_excel("all_evals_v5_evaluated.xlsx")
data.columns
criterias = list(set(data["criteria"]))
len(criterias)
checklists = []
for crit in criterias:
    val = ""
    try: 
        val = turnIntoChecklist(crit)
        print(val)
    except Exception as e:
        print(e)
    checklists.append(val)


#%%


evals_list =[]
for i , row in data.iterrows():
    print(i)
    print(row["letter_key"])
    key = row["letter_key"]
    letter = row["letterwrongdiagnosis"]
    try:
        out = correctDiagVsNot(letter, model="gpt-4-0613")
    except: 
        continue
    out = out["diagnosis_accuracy"]
    
    row_dict = {
        "original_key":key,
        "letter": letter,
        "stated_diagnosis":out["stated_diagnosis"],
        "actual_diagnosis":out["actual_diagnosis"],
        "diagonsis_correctness":out["diagnosis_correctness"],
        "comment": out["comment"]
        }

    df = pd.DataFrame([row_dict])
    evals_list.append(df)



wrong_letters=pd.concat(evals_list)
wrong_letters["diagonsis_correctness"]
#wrong_letters.to_csv("wrong_letters_diagnosis_correctness.csv")


#%% Medical report 

data = pd.read_excel("letters_with_wrong_diagnosis.xlsx")
evals_list =[]
for i , row in data.iterrows():
    print(i)
    print(row["letter_key"])
    key = row["letter_key"]
    letter = row["letter"]
    try:
        out = selectGuideline(letter, model="gpt-4-0613")
    except: 
        continue
    print(out["guideline"])
    row_dict = {
        "original_key":key,
        "letter": letter,
        "leading_symptom":out["leading_symptom"],
        "diagnosis":out["diagnosis"],
        "guideline":out["guideline"],
        }

    df = pd.DataFrame([row_dict])
    evals_list.append(df)

allx =pd.concat(evals_list)
allx
allx["guideline"]
#allx.to_excel("allx.xlsx")


#%% 



