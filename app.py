import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import LLMChain



from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="Movies Recommendation System With LLM")
st.title("Movies Recommendation System With LLM")

groq_api_key=st.sidebar.text_input(label="Groq API Key",type="password")

if not groq_api_key:
    st.info("Please enter Groq API Key to continue")
    st.stop()

llm=ChatGroq(model="llama-3.1-8b-instant",api_key=groq_api_key,temperature=0.7)


prompt="""You are a intelligent movies recommendation assistant.
Your task is to recommend movies or web series based on the user's current mood.
Understand the user mood then recommend 4-5 movies or web series according to high IMBD rating.
provide: Title,Genre,IMBD Rating, Why it matches the user's mood.
Mix moviies of web series from hollywood,bollywodd,anime,Korean.
User mood:{user_mood}
"""
prompt_template=PromptTemplate(input_variables=['user_mood'],template=prompt)

chain=LLMChain(llm=llm,prompt=prompt_template)

if 'messages' not in st.session_state:
    st.session_state['messages']=[
        {"role":"assistant","content":"Hi I'm movies or web series recommendation system. According to your current mood."}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

user_mood=st.text_area("Enter your mood",placeholder="Examples: Happy, sad, stressed, romantic")

if st.button("Get Recommendation"):
    if user_mood:
        with st.spinner("Finding best movies for you.."):
            st.session_state.messages.append({"role":"user","content":user_mood})
            st.chat_message("user").write(user_mood)
            response=chain.invoke({"user_mood":user_mood})
            movies_response=response["text"]
            st.session_state.messages.append({"role":"assistant","content":movies_response})
            st.chat_message("assistant").write(movies_response)
    else:
        st.warning("Please enter your mood.")
        


