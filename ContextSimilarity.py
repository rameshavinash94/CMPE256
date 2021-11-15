
from sentence_transformers import SentenceTransformer, util
import spacy
import pandas as pd
from operator import itemgetter

class ContextSimilarity:
    def __init__(self,model):
        self.model=model
        self.SimilarityScore=[]
        self.query=''

    def ContextSimilarity(self,query,contexts):
        #add doc1 to nlp1 object
        self.query=query
        Doc_1 = self.model.encode(query, convert_to_tensor=True)
       
        for context in contexts:
            Doc_2 = self.model.encode(contexts, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(Doc_1, Doc_2)
            similarity_rate = "{0}".format(cos_scores[0][0])
            self.SimilarityScore.append([context,similarity_rate])
        return self.SimilarityScore

    def SortSimilarity(self,by='desc'):
        if by=='desc':
            return sorted(self.SimilarityScore, key=itemgetter(1), reverse=True)
        else:
            return sorted(self.SimilarityScore, key=itemgetter(1))

    def TopNSimilarity(self,top_n=10):
        return sorted(self.SimilarityScore, key=itemgetter(1), reverse=True)[:top_n]

    def ConvertToDf(self,values):
        FinalDf = pd.DataFrame(values,columns=['Context','Similarity'])
        return FinalDf

    def MergeDf(self,df1,df2):
        Final_Df = df1.merge(df2,left_index=True, right_index=True)
        Final_Df.drop(columns=['Wikipedia_Paragraphs'],inplace=True)
        Final_Df.reset_index(inplace=True)
        Final_Df.drop(columns=['index'],inplace=True)
        return Final_Df

    def TopNSimilarityDf(self,Df,top_n=10):
        return Df.sort_values(by=['Similarity'], ascending=False).iloc[:top_n,]
