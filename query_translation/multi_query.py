from langchain_core.load import dumps, loads
from query_translation.base_multi_query import BaseMultiQuery


class MultiQuery(BaseMultiQuery):
    
    def generate_multi_queries(self):

        # Multi Query: Different Perspectives
        template = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""
        
        return self.generate_multiple_queries(template)
    
    def get_unique_union(self, documents: list[list]):
        """ Unique union of retrieved docs """
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]