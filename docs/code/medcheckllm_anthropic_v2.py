# -*- coding: utf-8 -*-
"""
"""
import os
import pandas as pd

import openpyxl

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

import json 


#%% Set Up


api_key = ""
llm = ChatAnthropic(model = "claude-3-sonnet-20240229", temperature=0.2, max_tokens=1024, api_key =api_key)

#%% Functions 

def extractDiagnosis(KEYS,report, model="claude-3-sonnet-20240229"):
   
    prompt = ChatPromptTemplate.from_messages([
       ( "user", "You are supposed to extract a diagnosis of a medical report. Of the choices, return the approriate headache type key for the diagnosis in the following medical report:\n\nKeys:{KEYS}\nMedical Report:\n{report}n\n Please return ONLY the key, so that I can access the dictionary directly., e.g.:\n14.1 Headache not elsewhere classified")
       ])
    chain = prompt | llm
    out = chain.invoke({"KEYS": KEYS, 
                  "report": report
                  })
    return out.content
KEYS= "'1.1 Migraine without aura', '1.3 Chronic migraine', '1.5 Probable migraine', '2.1 Infrequent episodic tension-type headache', '3.1 Cluster headache', '3.2 Paroxysmal hemicrania', '3.3 Short-lasting unilateral neuralgiform headache attacks', '3.4 Hemicrania continua', '3.5 Probable trigeminal autonomic cephalalgia', '4.1 Primary cough headache', '4.2 Primary exercise headache', '4.3 Primary headache associated with sexual activity', '4.4 Primary thunderclap headache', '4.7 Primary stabbing headache', '4.8 Nummular headache', '4.9 Hypnic headache', '4.10 New daily persistent headache (NDPH)', '5.1 Acute headache attributed to traumatic injury to the head', '5.2 Persistent headache attributed to traumatic injury to the head', '5.3 Acute headache attributed to whiplash1', '5.4 Persistent headache attributed to whiplash', '5.5 Acute headache attributed to craniotomy', '5.6 Persistent headache attributed to craniotomy', '6.4 Headache attributed to arteritis', '6.5 Headache attributed to cervical carotid or vertebral artery disorder', '6.9 Headache attributed to pituitary apoplexy', '7.1 Headache attributed to increased cerebrospinal fluid (CSF) pressure', '7.2  Headache attributed to low cerebrospinal fluid (CSF) pressure', '7.3 Headache attributed to non-infectious inflammatory intracranial disease', '7.4 Headache attributed to intracranial neoplasia', '7.5 Headache attributed to intrathecal injection', '7.6 Headache attributed to epileptic seizure', '7.7 Headache attributed to Chiari malformation type I (CM1)', '7.8 Headache attributed to other non-vascular intracranial disorder', '8.2 Medication-overuse headache (MOH)', '10.1 Headache attributed to hypoxia and/or hypercapnia', '10.2 Dialysis headache', '10.4 Headache attributed to hypothyroidism', '10.5 Headache attributed to fasting', '10.6 Cardiac cephalalgia', '10.7 Headache attributed to other disorder of homoeostasis', '11.1 Headache attributed to disorder of cranial bone', '11.4 Headache attributed to disorder of the ears', '11.6 Headache attributed to disorder of the teeth', '11.7 Headache attributed to temporomandibular disorder (TMD)', '11.8 Headache or facial pain attributed to inflammation of the stylohyoid ligament', '11.9 Headache or facial pain attributed to other disorder of cranium, neck, eyes, ears, nose, sinuses, teeth, mouth or other facial or cervical structure', '12.1 Headache attributed to somatization disorder1', '12.2 Headache attributed to psychotic disorder', '13.1 Pain attributed to a lesion or disease of the trigeminal nerve', '13.4 Occipital neuralgia', '13.5 Neck-tongue syndrome', '13.6 Painful optic neuritis', '13.7 Headache attributed to ischaemic ocular motor nerve palsy', '13.8 Tolosa-Hunt syndrome', '13.9 Paratrigeminal oculosympathetic (Raeder’s) syndrome', '13.10 Recurrent painful ophthalmoplegic neuropathy', '13.11 Burning mouth syndrome (BMS)', '13.12 Persistent idiopathic facial pain (PIFP)', '14.1 Headache not elsewhere classified', '14.2 Headache unspecified'"

class give_leadingsymdiagguideline(BaseModel):
    """give_leadingsymdiagguideline
    Get the leading symptom, diagnosis and medical guideline.
    - The leading symptom
    - The diagnosis
    - A relevant medical guideline for further evaluation

    """
    leading_symptom: str = Field(..., description = "leading symptom")
    diagnosis : str = Field(..., description = "diagnosis")
    guideline : str = Field(..., description = "A relevant medical guideline for further evaluation")


def selectGuideline(report, model="claude-3-sonnet-20240229"):
    
    prompt = ChatPromptTemplate.from_messages([
        ("user", """Please examine the following medical report and provide: - The leading symptom
        - potential syndrome 
        - The diagnosis
        - The name of a relevant, established medical guideline for the leading symptom.

\n Do not return actual medical advice but the name of a relevant medical guideline. Use the tool give_leadingsymdiagguideline

\nMedical Report:\n{report}\n\n"""
         
         )])
    llm_GuidelineSuggestion = llm.bind_tools([give_leadingsymdiagguideline])
    chain = prompt | llm_GuidelineSuggestion 
    output = chain.invoke({"report": report})
    return output


def turnIntoChecklist(guideline_out, model="claude-3-sonnet-20240229"):
    prompt = ChatPromptTemplate.from_messages([(
        "user", '''
         I will provide you with part of a medical guideline. If it is in a format that could be used as a checklist, return the word 'CHECKLIST'. Only return the word 'CHECKLIST', not the actual checklist itself.
         If it is a continuous, long text with complete sentences, try to extract a checklist from the guideline and return your checklist.
         \n
         Guideline:\n
             {guideline_out} 
         '''
        )])
    chain = prompt | llm

    output = chain.invoke({"guideline_out": guideline_out})
    return output

class give_output_of_diagnosis_evaluation(BaseModel):
    """give_output_of_diagnosis_evaluation
     
    """
    stated_diagnosis :  str = Field(..., description = "The diagnosis stated in the doctor's letter")
    actual_diagnosis : str = Field(..., description = "The diagnosis that you believe the patient actually has")
    diagnosis_correctness : str = Field(..., description = " whether the stated and actual diagnosis are the same (yes/no)")
    comment : str = Field(..., description = "Additional comments.")

def correctDiagVsNot(letter, model="claude-3-sonnet-20240229"):
    
    prompt = ChatPromptTemplate.from_messages([
        ("user","""
           I will provide you with a medical report. 
           Please assess whether the doctor's letter identified the correct diagnosis.
            Return the following results:
               - The diagnosis stated in the doctor's letter
               - The diagnosis that you believe the patient actually has
               - whether the stated and actual diagnosis are the same (yes/no)
               - Additional comments.
         Please use the tool give_output_of_diagnosis_evaluation.
         """ ), 
         ("user", "Letter:\n{letter}")
        ])
         
    llm_withDiagnosisEval = llm.bind_tools([give_output_of_diagnosis_evaluation])
    chain = prompt | llm_withDiagnosisEval
    output = chain.invoke({"letter": letter})
    return output
         


class ItemDetail(BaseModel):
    '''Information about each checklist item'''
    item: str = Field(description = "item")
    comprehensiveness : int = Field(description = "Comprehensiveness of coverage of checklist item: (0-5)")
    comments:  str=  Field(description ="Comments on checklist item")
      

class give_output_of_evaluation(BaseModel):
    """
    Output List 
    give_output_of_evaluation
    - item
    - Comments on checklist item (string)
    - Additional comments
    
    """
    checklist_completion_status : List[ItemDetail] = Field(description ="List of checklist items, each containing item,comprehensiveness,comment " )
    

def evaluateLetter(letter, checklist, diagnosis, model = "claude-3-sonnet-20240229"):
    prompt = ChatPromptTemplate.from_messages([
        ("user",  """
           I will provide you with a checklist guideline and a medical report. 
           Please assess the doctor's letter based on the checklist and consider the following: 
               Checklist items are usually numbered  A,B,C,D,E ... 
               Understand what is meant by for example: Any headache fulfilling criterion B and C.
               Then assess the letter and  
               
               return your results: 
             
           - For each checklist item separatedly: 
               - Whether the checklist item was thoroughly addressed in the doctor's letter (0-5: not covered at all (0) - comprehensively covered (5)).
               - Comment.
           Please use the give_output_of_evaluation tool.
         """
         ), 
            ("user", "Checklist for {diagnosis}:\n{checklist}\n---\nLetter:\n{letter}")
        
        ])
    llm_withChecklistEvaluation = llm.bind_tools([give_output_of_evaluation])
    chain = prompt | llm_withChecklistEvaluation
    output = chain.invoke({"letter": letter, 
                                                 "checklist": checklist,
                                                 "diagnosis": diagnosis
                                                 })
    return output


#%%
### Extraction of diagnosis
os.chdir("Y:\\Marc\\32.LLM\\MedCheckLLM\\Revision_V2\\Data\\")
data = pd.read_excel("letters_v2.xlsx")
data['Letter'] = data['Letter'].astype(str).apply(openpyxl.utils.escape.unescape)

evals_list =[]
for i , row in data.iterrows():
    print(i)
    #print(row["Letter"])
    letter_text = row["Letter"]
    letter_key = row["letter_key"]
    diagnosis = extractDiagnosis(KEYS,report = letter_text)

    letter_dict = {"letter_key":letter_key,
                   "Letter_text": letter_text, 
                   "identified_diagnosis": diagnosis
                   }
    df = pd.DataFrame([letter_dict])
    evals_list.append(df)

    
evaluated_df =pd.concat(evals_list)

#evaluated_df.to_excel("Letter_ExtractingStatedDiagnosis.xlsx")



#%%

# Suggesting of existing guidelines 

os.chdir("Y:\\Marc\\32.LLM\\MedCheckLLM\\Revision_V2\\Data\\")
data = pd.read_excel("letters_v2.xlsx")
data['Letter'] = data['Letter'].astype(str).apply(openpyxl.utils.escape.unescape)

evals_list =[]
for i , row in data.iterrows():
    print(i)
    #print(row["Letter"])
    letter_text = row["Letter"]
    letter_key = row["letter_key"]
    output = selectGuideline(report = letter_text)
    
    try:
        l_symptom = output.tool_calls[0]["args"]["leading_symptom"]
        diagnosis = output.tool_calls[0]["args"]["diagnosis"]
        guideline = output.tool_calls[0]["args"]["guideline"]
        
        letter_dict = {"letter_key":letter_key,
                       "Letter_text": letter_text, 
                       "l_symptom": l_symptom,
                       "diagnosis": diagnosis,
                       "guideline": guideline
                       }
    except Exception as e:
        print(str(e))
        letter_dict = {"letter_key":letter_key,
                       "Letter_text": letter_text, 
                       "l_symptom": "error",
                       "diagnosis": "error",
                       "guideline": str(e)
                       }

    
    df = pd.DataFrame([letter_dict])
    evals_list.append(df)

evaluated_df =pd.concat(evals_list)

#evaluated_df.to_excel("Letter_SuggestedGuidelines_Claude.xlsx")

#%%

# Detection of checklist 

data = pd.read_excel("analysis_table_prepClaude.xlsx")

criterias = list(set(data["criteria"]))
len(criterias)
checklists = []
for crit in criterias:
    print(crit)
    val = ""
    try: 
        val = turnIntoChecklist(crit)
        print(val.content)
    except Exception as e:
        print(str(e))
    checklists.append(val.content)
    
for idx, item in enumerate(checklists):
    print(idx, item)


#pd.DataFrame(checklists).to_excel("Turn_Into_Checklist.xlsx")

#%%

# Diagnostic Evaluation Correct vs Incorrect
# Correct letters

data = pd.read_excel("letters_with_wrong_diagnosis.xlsx") # contains also correct letters

evals_list =[]
for i , row in data.iterrows():
    print(i)
    print(row["letter_key"])
    key = row["letter_key"]
    letter = row["letter"]
    try:
        out = correctDiagVsNot(letter)
       
            
        row_dict = {
            "original_key":key,
            "letter": letter,
            "stated_diagnosis": out.tool_calls[0]["args"]["stated_diagnosis"],
            "actual_diagnosis": out.tool_calls[0]["args"]["actual_diagnosis"],
            "diagonsis_correctness":out.tool_calls[0]["args"]["diagnosis_correctness"],
            "comment": out.tool_calls[0]["args"]["comment"]
            }

    except Exception as e:
        print(str(e))
        continue

    df = pd.DataFrame([row_dict])
    evals_list.append(df)


evals_df = pd.concat(evals_list)
#evals_df.to_excel("CorrectVsNot_CorrectLetters.xlsx")


# Incorrect letters 
data = pd.read_excel("letters_with_wrong_diagnosis.xlsx") # contains also correct letters
data.columns


evals_list =[]
for i , row in data.iterrows():
    print(i)
    print(row["letter_key"])
    key = row["letter_key"]
    letter = row["letterwrongdiagnosis"]
    try:
        out = correctDiagVsNot(letter)
       
            
        row_dict = {
            "original_key":key,
            "letter": letter,
            "stated_diagnosis": out.tool_calls[0]["args"]["stated_diagnosis"],
            "actual_diagnosis": out.tool_calls[0]["args"]["actual_diagnosis"],
            "diagonsis_correctness":out.tool_calls[0]["args"]["diagnosis_correctness"],
            "comment": out.tool_calls[0]["args"]["comment"]
            }

    except Exception as e:
        print(str(e))
        continue

    df = pd.DataFrame([row_dict])
    evals_list.append(df)


evals_df = pd.concat(evals_list)
#evals_df.to_excel("CorrectVsNot_WrongLetters.xlsx")


#%%

# Evaluation of diagnostic criteria 

data = pd.read_excel("analysis_table_prepClaude_Input.xlsx")
eval_lists = []

for idx, row in data.iterrows():
    key = row["letter_key"]
    criteria = row["criteria"]
    letter_text = row["letter"]
    print(idx, key)

    try: 
        print("Criteria extracted. Now evaluating..")
        evals = evaluateLetter(letter = letter_text , checklist=criteria,  diagnosis=key)
        
        print("Evaluation done. Now saving..")
        evaluations = evals.tool_calls[0]["args"]["checklist_completion_status"]
        df_list =[]
        #len(evals.tool_calls)
        #len(evals.content)
        for checklist_item in evaluations:
            item = checklist_item["item"]
            comp = checklist_item["comprehensiveness"]
            comm = checklist_item["comments"]
            print(item, "\n", comp, "\n",comm)
            row_dict = {"letter_key": key,
                "identified_diagnosis":key, 
                        "criteria":criteria,
                           "letter": letter_text, 
                           "evals": json.dumps( evals.tool_calls[0]["args"]), 
                           "item": item,
                           "comprehensiveness": comp,
                           "comment": comm
                           }
            df = pd.DataFrame([row_dict])
            df_list.append(df)
        bound = pd.concat(df_list)
        bound.to_excel("analyzed_claude_" + key + ".xlsx")
        eval_lists.append(bound)
        print("##########################################Worked!")
    except Exception as e:
        print("ERROR -------------------------------------")
        print(str(e))
        eval_lists.append(str(e))
    

all_criteria_evaluated = pd.concat(eval_lists)
#all_criteria_evaluated.to_excel("analysis_table_Claude_criteria.xlsx")

evaluated = pd.read_excel("analysis_table_Claude_criteria_evaluated.xlsx")
evaluated["assessment"].value_counts()

