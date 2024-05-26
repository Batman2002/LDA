import streamlit as st
import os
from master import process_pdf,get_images,clean_lda_output,extract_abstract_and_references,convert_string_to_df,image_similarity,remove_similar_images,resize_image,apply_LDA,lemmatization,process_data,tokenize_data,create_bi_trigram,create_corpus,calculate_coherence
import pyLDAvis
from navigation import logout
import mysql.connector as con
import io
import gzip
from db_trial import insert_pdf_into_database,connect_to_database,retrieve_data,get_user_id


st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)


mydb=connect_to_database()
cursor=mydb.cursor()

# st.title(st.session_state.username)
st.title("Topic Modelling using LDA")
# print(index.uname)
if st.button("Logout"):
    logout()

uploaded_file=st.file_uploader("Upload your research paper:",type="pdf")



def show_plot(model,corpus,id2word):
    # pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim_models.prepare(model, corpus, id2word)
    html=pyLDAvis.prepared_data_to_html(vis)
    st.components.v1.html(html, width=1300, height=800)
    return html



if uploaded_file is not None:
    pdf_bytes=uploaded_file.read()

    text=process_pdf(uploaded_file)
    
    final_text=text.split('.')
    
    dataframe=extract_abstract_and_references(final_text)
    
    new_data=process_data(dataframe)
    
    sentences=tokenize_data(new_data)
    
    bag_of_words=create_bi_trigram(sentences)
    # texts=lemmatization(bag_of_words)
    
    corpus,id2word=create_corpus(bag_of_words)
    
    num_topics=st.slider("Enter the number of topics(default=10):",min_value=2,max_value=20,value=10)
    
    model=apply_LDA(corpus,id2word,num_topics)
    # show_plot(model,corpus,id2word)
    
    st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.write("### PyLDAvis Visualization:")
        pyldavis_html=show_plot(model, corpus, id2word)
        # st.write(pyldavis_html)
    
    
    with col2:
        st.write("")
    
    
    with col3:
        st.write("")
    
    
    insert_pdf_into_database(uploaded_file.name, pdf_bytes,pyldavis_html,mydb,get_user_id(st.session_state.username,mydb))
    st.write("### Topics:")
    topics=clean_lda_output(model.print_topics())
    i=1
    for topic in topics:
        st.write("Topic",i,":",(" ".join(topic)))
        i+=1
    # coherence=calculate_coherence(model,bag_of_words,id2word)
    # print('666666666666666666666666666666669999999999999999999999999999')
    # print(coherence)
    # print('666666666666666666666666666666669999999999999999999999999999')
    # st.write("### Coherence Score:",coherence)
else:
    st.subheader("Your data")
    data=retrieve_data(mydb,get_user_id(st.session_state.username,mydb))

    # st.write(data)

    for i in range(len(data)):
        st.write(data[i][2])
        st.components.v1.html(data[i][4], width=1300, height=800)
