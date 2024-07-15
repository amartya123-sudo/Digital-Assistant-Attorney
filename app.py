import os
import re
import streamlit as st
from dpr.main import DPR
from MultiDocQA.main import RAG
from MultiDocQA import redirect as rd

st.set_page_config(page_title="Digital Assistant Attorney",layout="wide")

hide_menu_style = """
<style>
header {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.header("Digital Assistant Attorney")

tab1, tab2 = st.tabs(["Legal-QA using DPR(Dense Passage Retriever)", "Multi-Document QA using RAG(Retrieval-Augmented Generation)"])

with tab1:
    st.header("Legal-QA using DPR(Dense Passage Retriever)")

    files = []
    for file in os.listdir("./dpr/data_MV"):
        if os.path.isfile(os.path.join("./dpr/data_MV",file)):
            fname=os.path.splitext(file)[0]
            files.append(fname)

    document = st.selectbox(label="Judgement Orders", options=files,index=None, placeholder="Select Judgement Order for QA")

    if document:
        text = open("./dpr/data_MV/"+document+".txt", 'r',encoding='utf-8').read()
        st.text_area(label="Source", value=text, height=300)

        question=st.text_input(label="Question")

        if question:
            with st.spinner("Thinking..."):
                dpr = DPR(context=text,question=question)
                st.write(dpr())

with tab2:
    st.header('Multi-Document QA using RAG(Retrieval-Augmented Generation)')
    st.subheader("Document Headings and Descriptions")

    for i in range(len(RAG.names)):
        st.subheader(f"{i + 1}) " + RAG.names[i])
        st.write(RAG.descriptions[i])

    def remove_formatting(output):
        output = re.sub('\[[0-9;m]+', '', output)  
        output = re.sub('\', '', output) 
        return output.strip()

    # @st.cache_resource
    def run(query):
        if query:
            with rd.stdout() as out:
                ox = RAG.processing_agent(query=query)
                print(ox)
            output = out.getvalue()
            output = remove_formatting(output)
            st.write(ox.response)
            return True
        
    query = st.text_input('Enter your Query.', key = 'query_input')
    ack = run(query)
    if ack:
        ack = False
        query = st.text_input('Enter your Query.', key = 'new_query_input')
        ack = run(query)
        if ack:
            ack = False
            query = st.text_input('Enter your Query.', key = 'new_query_input1')
            ack = run(query)
            if ack:
                ack = False
                query = st.text_input('Enter your Query.', key = 'new_query_input2')
                ack = run(query)
                if ack:
                    ack = False
                    query = st.text_input('Enter your Query.', key = 'new_query_input3')
                    ack = run(query)
                    if ack:
                        ack = False
                        query = st.text_input('Enter your Query.', key = 'new_query_input4')
                        ack = run(query)
                        if ack:
                            ack = False
                            query = st.text_input('Enter your Query.', key = 'new_query_input5')
                            ack = run(query)