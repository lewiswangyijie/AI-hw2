import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from openai import AzureOpenAI
import gradio as gra


class hw2:
    def __init__(self):
        self.load_env()
        self.db, self.new_db = self.load_data()

    def load_env(self):
        env_path = os.getenv("HOME") + "/Documents/src/openai/.env"
        load_dotenv(dotenv_path=env_path, verbose=True)

        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = "2023-05-15"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://pvg-azure-openai-uk-south.openai.azure.com"

        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-05-15"
        )

    def load_data(self):
        db_path = "data/irm-help"
        input_path = "data/IRM-Help.pdf"

        loader = PyPDFLoader(file_path=input_path)
        data = loader.load()

        db = FAISS.from_documents(data, AzureOpenAIEmbeddings())

        db.save_local(db_path)
        new_db = FAISS.load_local(db_path, AzureOpenAIEmbeddings())
        return db, new_db

    def search(self, query, threshold=0.7):
        answer_list = self.new_db.similarity_search(query)
        for ans in answer_list:
            print(ans.page_content + "\n")

        retriever = self.db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": threshold}
        )
        docs = retriever.get_relevant_documents("Managing Return Reasons")
        res = ""
        for doc in docs:
            res = res + (doc.page_content + "\n")
        return res

    def qa(self, query, threshold=0.7):
        llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(llm,
                                               retriever=self.new_db.as_retriever(
                                                   search_type="similarity_score_threshold",
                                                   search_kwargs={"score_threshold": threshold}))
        qa_chain.combine_documents_chain.verbose = True
        qa_chain.return_source_documents = True

        res = qa_chain.invoke({"query": query})
        return res["result"]


if __name__ == '__main__':
    hw2 = hw2()
    # hw2.search("Managing Return Reasons")
    # res = hw2.qa("Managing Return Reasons")
    # print(res)
    # app = gra.Interface(fn=hw2.qa, inputs=["text", "number"], outputs=["text"])
    with gra.Blocks() as demo:
        query = gra.Text(label="query")
        threshold = gra.Number(label="threshold")
        answer = gra.Text(label="answer")
        qa = gra.Button("qa")
        search = gra.Button("search")
        qa.click(hw2.qa, inputs=[query, threshold], outputs=answer)
        search.click(hw2.search, inputs=[query, threshold], outputs=answer)
    demo.launch(share=True)
