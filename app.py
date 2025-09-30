import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq  # Groq LLM wrapper

# Function to get response from Groq
def getGroqResponse(input_text, no_words, blog_style):
    # Initialize Groq LLM
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",       # latest Groq-supported LLaMA 3 model
        temperature=0.01,
        max_tokens=512
    )

    # Prompt Template
    template = """
    Write a blog for {blog_style} job profile
    on the topic "{input_text}"
    within {no_words} words.
    """

    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template
    )

    # Generate response
    response = llm.predict(prompt.format(
        blog_style=blog_style,
        input_text=input_text,
        no_words=no_words
    ))
    return response


# Streamlit UI
st.set_page_config(
    page_title="Generate Blogs",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("No of Words")
with col2:
    blog_style = st.selectbox(
        "Writing the blog for",
        ("Researchers", "Data Scientist", "Common People"),
        index=0
    )

submit = st.button("Generate")

if submit:
    if not input_text or not no_words:
        st.warning("⚠️ Please fill all fields before generating.")
    else:
        blog = getGroqResponse(input_text, no_words, blog_style)
        st.subheader("Generated Blog ✍️")
        st.write(blog)
