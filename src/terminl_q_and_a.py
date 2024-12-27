import yaml
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.load_config import LoadConfig
from typing import List, Tuple

# Load credentials 
APPCFG = LoadConfig()

with open("config/app_config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)

# Load the embedding function
embeddings = APPCFG.embedding_model

# Load the DB
vectordb = Chroma(persist_directory=APPCFG.persist_directory,
                  embedding_function=embeddings)
#Terminal QA
while True:
    question = input("\n\nEnter your question or press 'q' to exit: ")
    if question.lower() == 'q':
        break
    question = '# user new question:\n' + question
    docs = vectordb.similarity_search(question, k=APPCFG.k)
    retrieved_docs_page_content: List[Tuple] = [
        str(x.page_content)+"\n\n" for x in docs]
    retrieved_docs_str = "# Retrieved content:\n\n" + str(retrieved_docs_page_content)
    prompt = retrieved_docs_str + "\n\n" + question
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature = 0
    )
    messages = [
        ("system", APPCFG.llm_system_role),
        ("human", prompt)
    ]
    response = llm.invoke(messages)
    print(response.content)
