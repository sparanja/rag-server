from llm import LLM
from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class BaseMultiQuery:

    def generate_multiple_queries(self, template: str):
        prompt_perspectives = ChatPromptTemplate.from_template(template)


        generate_queries = (
            prompt_perspectives 
            | LLM.get_instance().get_chat_openai()
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

        return generate_queries