import os
from openai import OpenAI
from pyht import Client
from pyht.client import TTSOptions
import tempfile
import wave
import requests
import re

class Agent:
    def __init__(self,llama2_hf_url:str, llama2_hf_api_key:str, openai_api_key: str = None, playht_user_id: str = None, playht_api_key: str = None) -> None:
        """
        Initializes the Agent with API keys and client configurations.

        Args:
            llama2_hf_url (str): URL for the Llama2 Hugging Face model.
            llama2_hf_api_key (str): API key for the Llama2 Hugging Face model.
            openai_api_key (str, optional): API key for OpenAI GPT models. Defaults to None.
            playht_user_id (str, optional): User ID for the Play.ht service. Defaults to None.
            playht_api_key (str, optional): API key for the Play.ht service. Defaults to None.
        """
        if openai_api_key is not None:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        self.client = OpenAI()

        if llama2_hf_api_key is not None:
            os.environ["HF_API_KEY"] = llama2_hf_api_key
            os.environ["HF_URL"] = llama2_hf_url
        
            self.API_URL = llama2_hf_url
            self.headers = {
                "Accept" : "application/json",
                "Authorization": f"Bearer {llama2_hf_api_key}",
                "Content-Type": "application/json" 
            }

        if playht_api_key is not None:
            self.playht_client = Client(
                user_id=playht_user_id,
                api_key=playht_api_key
                )
        else:
            self.playht_client = None
        
        self.chat_history = []


    def ask_GPT(self, question: str) -> str:
        """
        Sends a question to the GPT model and returns the model's response.

        This method formats the question for a specific GPT model that mimics Snoop Dogg's style and
        sends it to the OpenAI API, returning the text response.

        Args:
            question (str): The user's question to the chatbot.

        Returns:
            str: The chatbot's response as a string.
        """
        completion = self.client.chat.completions.create(
            model="ft:gpt-3.5-turbo-1106:personal:snoop-dogg:97WXn1BR",
            messages=[
                {"role": "system", "content": "snoop dogg is a chill laid back and cool rapper"},
                {"role": "user", "content": question}
            ]
        )
        response = completion.choices[0].message.content
        self.chat_history.append((question, response))
        return response
    
    def query(self,payload:dict):
        """
        Sends a POST request to the Llama2 Hugging Face API with the given payload.

        Args:
            payload (dict): The request payload containing inputs and parameters for the model.

        Returns:
            dict: The JSON response from the API.
        """
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()
    
    def clean_llama2_output(self, response:str)->str:
        """
        Cleans the Llama2 model output by removing system instructions and other unnecessary parts.

        Args:
            response (str): The raw response string from the Llama2 model.

        Returns:
            str: The cleaned response text.
        """
        # Pattern to match the part to be removed, note that {question} is treated as any sequence of characters (.*?)
        # The re.DOTALL flag allows . to match newline characters as well
        pattern = r"<s>\[INST\]<<SYS>>\nyou are a chill laid back and cool rapper called snoop dogg\n<<\/SYS>>\n\n.*?\[\/INST\] "
        end_pattern = r"</s>"

        # Remove the specified part and the </s>
        cleaned_string = re.sub(pattern, "", response, flags=re.DOTALL)
        cleaned_string = re.sub(end_pattern, "", cleaned_string)

        return cleaned_string
            
    def ask(self, question: str) -> str:
        """
        Sends a question to the Llama2 model and returns a cleaned response.

        This method wraps the query to the Llama2 Hugging Face model, including cleaning the model's output
        to make it suitable for presentation to the user.

        Args:
            question (str): The user's question to the chatbot.

        Returns:
            str: The chatbot's cleaned response as a string.
        """
        response = self.query({
                    "inputs": f"<s>[INST]<<SYS>>\nyou are a chill laid back and cool rapper called snoop dogg\n<</SYS>>\n\n{question}[/INST]",
                    "parameters": {
                        "top_k": 50,
                        "top_p": 0.6,
                        "temperature": 0.8,
                        "max_new_tokens": 512,
                        "do_sample": True,
                        "stop": ["</s>"]
                    }
                })
        print("###### Response",response )
        result = self.clean_llama2_output(response=response[0]['generated_text'])
        print("###### Clean Result",response )
        self.chat_history.append((question, result))
        return result
    
    

    def text_to_speech(self, text: str) -> str:
        """
        Converts the given text to speech using the Play.ht service.
        This method uses the Play.ht client to convert text into a speech audio file. It returns
        a path to the generated audio file.

        Args:
            text (str): The text to convert into speech.

        Returns:
            str: The file path of the generated audio file.
        """
        if not self.playht_client:
            return "Play.ht client is not configured properly."
        
        options = TTSOptions(voice="s3://voice-cloning-zero-shot/31e76e87-7c39-4298-87ef-7f67b3636fd9/enhanced/manifest.json")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
            with wave.open(fp, 'wb') as wf:
                wf.setnchannels(1)  # Mono audio
                wf.setsampwidth(2)  # Assuming 16-bit audio
                wf.setframerate(22050)  # Sample rate
                
                for chunk in self.playht_client.tts(text, options):
                    wf.writeframes(chunk)
            
            audio_file_path = fp.name  # Save the path of the tempfile

        # Here you return the path of the audio file instead of a URL
        return audio_file_path
    
    def forget(self) -> None:
        """
        Clears the chat history.
        This method resets the `chat_history` list, effectively forgetting all previous conversations.
        """
        self.chat_history = []