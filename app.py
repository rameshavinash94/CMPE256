
import spacy
import pandas as pd
import spacy_universal_sentence_encoder
import re
import streamlit as st
import requests
from ContextExtraction import ContextExtraction
from DocumentRetrival import DocumentRetrival
from DataWrangling import DataWrangler
from ContextSimilarity import ContextSimilarity
from MLModel import MLModel
from flatten_json import flatten
from spacy.lang.en import English


@st.cache(hash_funcs={spacy.lang.en.English:id})
def load_models():
  nlp = spacy.load('en_core_web_lg')
  use_nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')
  return nlp,use_nlp

nlp, use_nlp = load_models()

with st.form(key='my_form'):
    question = st.text_input('Type your query', 'who is mark zuckerberg?')
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
  #create a Document retrival object
  doc_retrive_obj = DocumentRetrival(nlp)

  #call UserInput func to get query input
  query = doc_retrive_obj.UserInput(question)

  #call preprocess func to preprocess query if required
  doc_retrive_obj.PreprocessUserInput()

  #call Retrive func with required top_n docs for retrival from Wiki
  pages = doc_retrive_obj.Retrive(3)

  #check if no pages are captured by the api
  if len(pages) == 0:
    st.error("kindly refine your Search, we are not able to find all relevant pages!!!!")

  #create a Extraction retrival object
  context_extract_obj = ContextExtraction(nlp)

  # Create a spacy matcher for the user query to parse the pages
  context_extract_obj.AddPhraseMatcher(query)

  # extract necessary context
  context_extract_obj.RetriveMatch(pages)

  # convert to pandas df
  text = context_extract_obj.StoreFindingAsDf()

  # store_results in csv for further reference
  #text.to_csv("Matching_Wiki_contexts.csv")

  #create a Data Wrangler object
  data_wrangler_obj = DataWrangler(nlp)

  #cleaned Dataframe
  cleaned_df = data_wrangler_obj.DataWranglerDf(text)

  # store_results in csv for further reference
  #cleaned_df.to_csv("Cleaned_Wiki_contexts.csv")

  #create a Context Similarity object
  context_similarity_obj = ContextSimilarity(use_nlp)

  #find the Similarites of Different context
  con_list = context_similarity_obj.ContextSimilarity(query,cleaned_df['Wikipedia_Paragraphs'])

  context_similarity_df = context_similarity_obj.ConvertToDf(con_list)

  Merged_Df = context_similarity_obj.MergeDf(context_similarity_df,cleaned_df)

  #retreive top N rows from dataframe
  TopNDf = context_similarity_obj.TopNSimilarityDf(Merged_Df,top_n=20)

  #create a ML Model object
  ML_Model_obj = MLModel()

  #call the Roberta model
  roberta_finding = ML_Model_obj.RobertaModel(TopNDf,query)

  #final Df post model prediction
  Final_DF = ML_Model_obj.ConverttoDf()

  #filtering only top N out of it.
  Results = ML_Model_obj.TopNDf(Final_DF,top_n=5)
  Results['Imageapi'] = 'https://en.wikipedia.org/w/api.php?action=query&titles='+ Results['Wiki_Page'].astype('str').str.extract(pat = "('.*')").replace("'", '', regex=True) + '&prop=pageimages&format=json&pithumbsize=100'
  Results['Wiki_Page'] = 'https://en.wikipedia.org/wiki/' + Results['Wiki_Page'].astype('str').str.extract(pat = "('.*')").replace("'", '', regex=True)
  #Results['Wiki_Page'] = Results['Wiki_Page'].replace(" ", '_', regex=True)
  #Results.to_csv('final_results.csv')
  for index, row in Results.iterrows():
    st.markdown('**{0}**'.format(row['Prediction'].upper()))
    r = requests.get(row['Imageapi'])
    test = r.json()
    flat_json = flatten(test)
    for x,y in flat_json.items():
      if re.findall('https.*',str(y)):
        st.image(y)
    st.markdown('_wiki:_ **{0}**'.format(row['Wiki_Page']))
    cont = '<p style="font-family:sans-serif; color:black; font-size: 8px;">{0}</p>'.format(row['Context'])
    st.write(cont,unsafe_allow_html=True)
