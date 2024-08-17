import os
import re
import openai
import json
import streamlit as st
from dpr.main import DPR
from MultiDocQA.main import RAG
from MultiDocQA import redirect as rd
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from kor import JSONEncoder

from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.query_engine import CitationQueryEngine

from engine.main import LegalCaseSearchApp


st.set_page_config(page_title="Digital Assistant Attorney",layout="wide")

hide_menu_style = """
<style>
header {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.header("Digital Assistant Attorney")

tab1, tab2, tab3, tab4 = st.tabs(["Legal Search Engine", "Legal-QA using DPR(Dense Passage Retriever)", "Multi-Document QA using RAG(Retrieval-Augmented Generation)", "AutoGPT"])

with tab1:
    st.title("Legal Case Search")

    app = LegalCaseSearchApp()

    query = st.text_input("Search")

    if st.button("Search"):
        results = app.combined_search(query)
        if results:
            case_details = app.get_case_details(results)
            for case in case_details:
                st.subheader(case['title'])
                st.write(f"Court Name: {case['court_name']}")
                st.write(case['content'])
        else:
            st.write("No matching cases found.")

with tab2:
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

with tab3:
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
    
with tab4:
    @st.cache_resource
    def preprocess_prelimnary():
        storage_context = StorageContext.from_defaults(persist_dir = './autogpt/persist')
        index = load_index_from_storage(storage_context = storage_context)
        query_engine = CitationQueryEngine.from_args(index, similarity_top_k = 1, citation_chunk_size = 512)

        return query_engine

    def load_docstore(filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data

    def find_node_text(docstore_data, target_node_id):
        for key, value in docstore_data['docstore/data'].items():
            relationships = value['__data__'].get('relationships', {})
            for rel_key, rel_value in relationships.items():
                if rel_value.get('node_id') == target_node_id:
                    return value['__data__'].get('text', "Text not found")
        return "Node text not found"

    client=openai.OpenAI()

    q_e = preprocess_prelimnary()
    docstore_data = load_docstore('./autogpt/persist/docstore.json')


    @st.cache_data
    def generate_petition(input_situation,model):
        # openai.ChatCompletion.create
        response = client.chat.completions.create(
                model = model,
                messages=[
                            {"role": "system", "content": "You are a helpful assistant who answers questions."},
                            {"role": "user", "content": f"create a sample petition for the situation {input_situation}. Use Indian laws."},
                ]
        )
        
        return response.choices[0].message.content



    @st.cache_data
    def get_approaches(_chain, input_situation, res_edited):
        try:
            approaches = _chain.run(res_edited).replace('<json>', '').replace('</json>', '')
            # approaches = json.loads(approaches)
        except Exception as e:
            st.write(f"Error: {e}")
            approaches = _chain.run(res_edited)
            # approaches = json.loads(approaches)
        with open('temp.json', 'w') as file:
            json.dump(approaches, file)
        return approaches

    input_situation = st.text_input(label = 'Create a sample petition for the situation -')
    model = st.selectbox(label = 'Please select a model -', options = ['gpt-3.5-turbo','gpt-4','gpt-3.5-turbo-0125','gpt-4-0613'])
    temperature = st.slider('Adjust temperature -', min_value = 0.0, max_value = 1.0, value = 0.0, step = 0.1)

    start = st.checkbox('Make Tree')

    llm = ChatOpenAI(
        model_name = model,
        temperature = temperature,
    )

    example_petition = """[Your Full Name]
    [Your Address]
    [City, State, Postal Code]
    [Email Address]
    [Phone Number]
    [Date]

    To,

    The Board of Directors
    [Company's Full Name]
    [Company's Address]
    [City, State, Postal Code]

    Subject: Petition challenging the proposed Corporate Restructuring Plan involving the Amalgamation of Subsidiaries

    Dear Members of the Board,

    I, [Your Full Name], a concerned shareholder of [Company's Full Name], wish to bring your attention to the proposed corporate restructuring plan involving the amalgamation of subsidiaries, announced on [Date of Announcement]. I am writing this petition to highlight the potential adverse impacts of this decision on the company's long-term sustainability, the stakeholders' interests, and to challenge the validity of this decision based on the Company's Act, 2013.

    While I understand that the decision was made with the intent to streamline operations and increase efficiency, it is important to evaluate the repercussions carefully. The amalgamation could potentially lead to job losses, reduced competition, and have a detrimental impact on consumers and small businesses.

    Under Section 394 of the Companies Act, 2013, I propose that an independent and thorough investigation be conducted to ensure that this restructuring plan serves the best interest of all shareholders, employees, creditors, and other stakeholders of [Company's Full Name].

    Furthermore, it is requested that the amalgamation proposal be tabled for discussion and voting at the upcoming general meeting. I believe this will enable a democratic process where shareholders can voice their opinions and concerns, as provided under Section 230(3) of the Companies Act, 2013.

    I urge the Board to consider a revised plan or alternative measures that could address the aforementioned issues. To this end, I propose the formation of a committee comprising representatives of shareholders, employees, and independent experts to review the proposed plan.

    If there is a failure to address these concerns, I, along with other like-minded shareholders, may be forced to seek legal remedies available to us under Section 245 (Class Action Suits) of the Companies Act, 2013, and other relevant statutes.

    I hope you will consider this petition seriously and look forward to engaging in constructive dialogue regarding the proposed restructuring plan.

    Yours Sincerely,

    [Your Full Name]
    [Your Designation - if applicable]
    [Signature]

    CC:

    Registrar of Companies, Ministry of Corporate Affairs, Government of India
    Securities and Exchange Board of India
    [Company's Full Name] Shareholders' Association
    Attachments:

    List of supporting shareholders and signatures
    Copies of any relevant documentation
    """

    # type_description = """
    # ```TypeScript

    # petition_tree: Array<{ // Has Tree Nodes inside
    #  title: string // Contains a short approach title, Node number, and the corresponding question we need to address.
    #  query_legal_database: string // What precedents do we need to look for.
    #  query_legal_acts: string // What legal acts do we need to look for.
    # }>
    # ```"""

    if start:
        res = generate_petition(input_situation, model)
        res_edited = st.text_area("Edit Area", res, height = 200)
        # edit_petition = st.button(label = 'If you want to edit the petition generated.')
        # if edit_petition:
        #     edited_data = st.text_area("Edit Area", res)
        #     finish_edit_petition = st.button(label = 'Click when done.')
        #     if finish_edit_petition:
        #         res = edit_petition
        instruction_template = PromptTemplate(
            input_variables = [],
            template=(
                f"[Considering the situation and the corresponding petition, create a potential tree of path that the opposing counsel might follow or take in order to defend against the petition. Do include questions that query a legal database to find relevant statues and precedents. Furthermore, frame questions that are non-trivial and answerable by referencing multiple acts and precedents. The questions to be asked against a legal database has to complex and must refer precedents and acts. The following has to be the structure of the tree: The tree has to have atleast five and maximum 100 branches. the branches will have multiple nodes. Under every node expand upon the potential points to argue upon.  The nodes and branches can overlap. There has to be multiple intersection branches. remember the nodes should be framed as a question or a comlex legal situation. The leaf node should end as a question. The question should be followed up by a legal research question referring to a legal database and legal acts database. The questions should be tailored to indian case laws and acts. ]\n\n"
                # "Add some type description\n\n"
                # "{type_description}\n\n" # Can comment out
                # "Add some format instructions\n\n"
                # f"{format_instructions}\n"
                # "Suffix heren\n"
            ),
        )
        schema = Object(
            id = "petition_tree",
            description = "Has Tree Nodes inside",
            examples = [
                (example_petition, [{"title" : "Legality of the Restructuring Plan - Node 1.1 - Was the decision to amalgamate made in accordance with the Company's Act, 2013?", 
                                    "query_legal_database" : "Search for precedent case laws where the Companies Act, 2013 has been interpreted in relation to corporate restructuring plans, specifically amalgamation of subsidiaries.", 
                                    "query_legal_acts" : "Cross-reference Section 230 and Section 232 of the Companies Act, 2013 to identify whether all procedural requirements and legal compliances for amalgamation have been followed."}, 
                                    {"title" : "Legality of the Restructuring Plan - Node 1.2 - Was the decision-making process transparent and was it properly communicated to all stakeholders?", 
                                    "query_legal_database" : "Search for precedent case laws involving transparency and communication of corporate decisions.", 
                                    "query_legal_acts" : "Refer to Section 118, 119, and 120 of the Companies Act, 2013 to assess compliance with the rules regarding minutes of meetings and disclosure of material information."}, 
                                    {"title" : "Shareholder's Interest - Node 2.1 - Has the potential dilution of shareholders' value due to the restructuring been duly considered and justified?", 
                                    "query_legal_database" : "Find case laws on shareholder rights during corporate restructuring or amalgamation.", 
                                    "query_legal_acts" : "Cross-reference Sections 230 to 240 of the Companies Act, 2013 that deal with compromises, arrangements and amalgamations."}, 
                                    {"title" : "Shareholder's Interest - Node 2.2 - Were shareholders given an appropriate chance to voice their concerns and vote on the restructuring plan?", 
                                    "query_legal_database" : "Look for precedent cases where shareholders' voting rights were infringed upon during major corporate decisions.", 
                                    "query_legal_acts" : "Refer to Sections 101 to 107 and Section 109 of the Companies Act, 2013 to ensure compliance with rules concerning general meetings and voting rights."}, 
                                    {"title" : "Impact on Stakeholders - Node 3.1 - Has there been a thorough evaluation of the plan's effect on employees and stakeholders?", 
                                    "query_legal_database" : "Find precedent cases where restructuring has been challenged based on its adverse impacts on stakeholders.", 
                                    "query_legal_acts" : "Refer to the Industrial Disputes Act, 1947, in conjunction with the Companies Act, 2013, to analyze the rights of employees in situations of corporate restructuring."},
                                    {"title" : "Impact on Stakeholders - Node 3.2 - Will the amalgamation lead to reduced competition or formation of monopoly?", 
                                    "query_legal_database" : "Search for case laws relating to competition law and amalgamation.", 
                                    "query_legal_acts" : "Cross-reference the Competition Act, 2002, and the Companies Act, 2013, to understand the intersection of corporate law and competition law in the context of amalgamation."}, 
                                    {"title" : "Procedure and Disclosure - Node 4.1 - Were the procedural requirements for the proposed amalgamation properly followed?", 
                                    "query_legal_database" : "Search for case laws where the procedure for amalgamation under the Companies Act, 2013, was challenged.", 
                                    "query_legal_acts" : "Cross-reference Sections 232 and 233 of the Companies Act, 2013, to ensure that the procedural requirements for the proposed amalgamation were duly followed."}, 
                                    {"title" : "Procedure and Disclosure - Node 4.2 - Was there any violation of disclosure norms in the course of this process?", 
                                    "query_legal_database" : "Search for case laws involving violation of disclosure norms in the context of corporate restructuring.", 
                                    "query_legal_acts" : "Refer to SEBI (Listing Obligations and Disclosure Requirements) Regulations, 2015 and the Companies Act, 2013 for the required norms for disclosure."},
                                    {"title" : "Legal Remedies - Node 5.1 - What legal remedies do the shareholders have if the restructuring is pushed through despite objections?", 
                                    "query_legal_database" : "Search for case laws where shareholders have invoked their rights against corporate decisions, especially under Section 245 (Class Action Suits) of the Companies Act, 2013.", 
                                    "query_legal_acts" : "Refer to Section 245 of the Companies Act, 2013, and analyze the provision's applicability in the current situation."}])
            ],
            attributes = [
                Text(
                    id = "title",
                    description = "Contains a short approach title, Node number, and the corresponding question we need to address.",
                ),
                Text(
                    id = "query_legal_database",
                    description = "What precedents do we need to look for.",
                ),
                Text(
                    id = "query_legal_acts",
                    description = "What legal acts do we need to look for.",
                )
            ],
            many = True,
        )
        
        chain= create_extraction_chain(llm, schema, instruction_template = instruction_template, encoder_or_encoder_class = JSONEncoder())
        start_approaches = st.checkbox('Formulate Approaches')

        if start_approaches:
            try:
                output = get_approaches(chain, input_situation, res_edited)
                st.json(output)
            except:
                st.write('Failed, retrying.')
                st.cache_data.clear()
                output = get_approaches(chain, input_situation, res_edited)
                st.json(output)
            edit_choice = st.checkbox('Do you want to edit the approaches.')

            if edit_choice:
                edited_data = st.text_area("Paste updated JSON here", json.dumps(output, indent = 4))
                try:
                    updated_data = json.loads(edited_data)
                    st.subheader("Updated JSON data")
                    st.json(updated_data)
                    with open('temp.json', 'w') as file:
                        json.dump(updated_data, file)
                    with open('temp.json', 'r') as file:
                        json_data = json.load(file)
                    
                except json.JSONDecodeError:
                    st.error("Invalid JSON format. Please enter valid JSON data.")
            else:
                with open('temp.json', 'r') as file:
                    json_data = json.load(file)

            get_legal_act_solutions = st.checkbox(label = 'Get the possible legal approaches for the legal actions mentioned in the act')
            
            petition_tree = json_data['data']['petition_tree']
            
            questions_a = [i['query_legal_acts'] for i in petition_tree]
            questions_d = [i['query_legal_database'] for i in petition_tree]        

            if get_legal_act_solutions:
                answers = []
                database_answers = []

                base_append = "\nList similar cases. If not exact, list similar cases. Extract the case title for each case."

                for n, q in enumerate(questions_a):
                    st.subheader(f"{q} -")
                    response = client.chat.completions.create(
                            model = model,
                            messages=[
                                        {"role": "system", "content": "You are a helpful assistant who answers questions."},
                                        {"role": "user", "content": f"Based on the petition - \n\"{res_edited}\"\n, a possible approach was created {q}. Just state the acts that can be used in the legal approach without any explanation of how the acts can help pr explaining them."},
                                ]
                            )
                    st.write(response.choices[0].message.content)
                    answers.append(response.choices[0].message.content)
                    st.subheader(f"{questions_d[n]} -")
                    database_answer = q_e.query(questions_d[n] + base_append)
                    
                    
                    for i in range(len(database_answer.source_nodes)):
                        node_id=database_answer.source_nodes[i].node.source_node.node_id
                        node_text=find_node_text(docstore_data, node_id)
                        if node_text:
                            split_text = node_text.split("1.", 1)  
                            if len(split_text) > 0:
                                # st.write("Case Name: **",split_text[0], "**") 
                                st.write(f"<b>Case Name:</b> <b>{split_text[0]}</b>", unsafe_allow_html=True)
    
                            else:
                                st.write("Node text does not contain '1.'")
                        else:
                            st.write("Node not found")
                            
                        st.write(database_answer.source_nodes[i].node.get_text())
                        # st.write(database_answer.source_nodes[i].node.source_node)
                    database_answers.append(database_answer.response)
                
                final_str = ""

                for a in answers:
                    final_str += a + '\n'
                response = client.chat.completions.create(
                            model = model,
                            messages=[
                                        {"role": "system", "content": "You are a helpful assistant who answers questions."},
                                        {"role": "user", "content": f"Create a neat bullet point in markdown for the following approaches - {final_str}"},
                                ]
                            )
                
                st.header('Approach Summarised -')
                st.subheader('Query Legal Acts Summarised -')
                st.markdown(response.choices[0].message.content)
                formatting = ''
                st.subheader('Query Legal Database Summarised -')
                for n, i in enumerate(database_answers):
                    formatting += '* ' + questions_d[n] + '\n' + i + '\n'
                st.markdown(formatting)