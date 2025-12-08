"""
LLM Singleton Class
Provides a singleton pattern implementation for ChatOpenAI to ensure only one instance is created.
"""

import os
from langchain_openai import ChatOpenAI


class LLM:
    """
    Singleton class for ChatOpenAI instance.
    Ensures only one instance of ChatOpenAI is created and reused throughout the application.
    
    Usage:
        # Get the singleton instance
        llm_instance = LLM.get_instance()
        
        # Get the ChatOpenAI instance
        chat_openai = llm_instance.get_chat_openai()
        
        # Or use it directly in chains
        llm = LLM.get_instance().get_chat_openai()
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """
        Create a new instance only if one doesn't exist.
        Returns the existing instance if it already exists.
        """
        if cls._instance is None:
            cls._instance = super(LLM, cls).__new__(cls)
            cls._instance._chat_openai = None
        return cls._instance
    
    def __init__(self):
        """
        Initialize the ChatOpenAI instance only once.
        """
        if not LLM._initialized:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key must be provided via OPENAI_API_KEY environment variable"
                )
            
            self._chat_openai = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=api_key
            )
            LLM._initialized = True
    
    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of LLM.
        
        Returns:
            LLM: The singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_chat_openai(self):
        """
        Get the ChatOpenAI instance.
        
        Returns:
            ChatOpenAI: The ChatOpenAI instance
        """
        if self._chat_openai is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key must be provided via OPENAI_API_KEY environment variable"
                )
            
            self._chat_openai = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=api_key
            )
        return self._chat_openai
    
    def __call__(self):
        """
        Make the instance callable to get the ChatOpenAI instance.
        
        Returns:
            ChatOpenAI: The ChatOpenAI instance
        """
        return self.get_chat_openai()
    
    def __getattr__(self, name):
        """
        Delegate attribute access to the ChatOpenAI instance.
        This allows using LLM instance as if it were the ChatOpenAI instance directly.
        
        Args:
            name: The attribute name to access
            
        Returns:
            The attribute value from the ChatOpenAI instance
        """
        return getattr(self.get_chat_openai(), name)

