import os
from dotenv import load_dotenv
import yaml
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pyprojroot import here
import shutil

load_dotenv()

class LoadConfig:
    def __init__(self) -> None:
        with open(here("config/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

            #LLM config
            self.llm_engine = app_config["llm_config"]["engine"]
            self.llm_system_role = app_config["llm_config"]["llm_system_role"]
            self.persist_directory = str(here(
                app_config["directories"]["persist_directory"]
            ))
            self.custom_persist_directory = str(here(
                app_config["directories"]["custom_persist_directory"]
            ))
            self.embedding_model = GoogleGenerativeAIEmbeddings()

            #Retrieval config
            self.data_directory = app_config["directories"]["data_directory"]
            self.k = app_config["retrieval_config"]["k"]
            self.embedding_model_engine = app_config["embedding_model_config"]["engine"]
            self.chunk_size = app_config["splitter_config"]["chunk_size"]
            self.chunk_overlap = app_config["splitter_config"]["chunk_overlap"]

            #Memory
            self.number_of_q_a_pairs = app_config["memory"]["number_of_q_a_pairs"]

            # Load OpenAI credentials
            self.load_openai_cfg()

            # clean up the upload doc vectordb if it exists
            self.create_directory(self.persist_directory)
            self.remove_directory(self.custom_persist_directory)

    def load_google_api_key(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set in the environment.")


    def create_directory(self, directory_path: str):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def remove_directory(self, directory_path: str):
        if os.path.exists(directory_path):
            try:
                shutil.rmtree(directory_path)
                print(
                    f"The directory '{directory_path}' has been successfully removed.")
            except OSError as e:
                print(f"Error: {e}")
        else:
            print(f"The directory '{directory_path}' does not exist.")
