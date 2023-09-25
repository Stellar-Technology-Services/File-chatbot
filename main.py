#importing ui libraries
import streamlit as st

import magic

#env libraries
import os
from dotenv import load_dotenv

#Library to handle pdfs
import PyPDF2

#langchain modules
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, PythonCodeTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings

mime=magic.Magic()

def response(query, file):
    text = ''

    llm=OpenAI(temperature=0.5, openai_api_key=openai_api_key)

    ftype=mime.from_buffer(file)

    if ftype=="application/pdf" or ftype=="PDF document":

        reader=PyPDF2.PdfFileReader(file)

        num_pages = reader.numPages
        for i in range(num_pages):
            text+=reader.getPage(i).extractText()

        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
        )

    elif ftype=="text/plain" or ftype=="ASCII text":

        text=file.read()
        text_splitter= PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=200)

    elif ftype=="text/html" or ftype=="HTML document":

        

    texts=text_splitter.split_text(text)
    embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
    doc=FAISS.from_texts(texts, embeddings)


    qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=doc.as_retriever())

    if query:
        response=qa.run(query)
    else:
        return " "

    return response.translate(str.maketrans("", "", "_*"))

#loading api key to env
openai_api_key=os.getenv("OPENAI_API_KEY")
load_dotenv()

st.set_page_config(page_title="PDF chatbot", layout="wide")

#header
st.header("PDF chatbot")

#upload pdf file

file=st.file_uploader("Upload a PDF file")

#enter the query
query=st.text_input("Enter your query:")

#response
st.title("Response")
if file is not None:
    st.write(response(query, file))


