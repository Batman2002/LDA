import PyPDF2
import pandas as pd
import tabula
from nltk.tokenize import sent_tokenize
from pdfminer.high_level import extract_pages,extract_text
import fitz
from PIL import Image
import io
import cv2

image_names_arr=[]


# this functions processes pdf will have the same path(pdf)
def process_pdf(path):
    # get_images(path)
    text=extract_text(path)
    return text

def get_images(path):
    pdf=fitz.open(path)
    counter=1
    for i in range(len(pdf)):
        page=pdf[i]
        images=page.get_images()
        for image in images:
            base_img=pdf.extract_image(image[0])
            image_data=base_img['image']
            img=Image.open(io.BytesIO(image_data))
            extension=base_img['ext']
            img.save(open(f'image{counter}.{extension}','wb'))
            image_names_arr.append(f'image{counter}.{extension}')
            counter+=1


# this function removes everything above abstract and everything below references(inclusive)
def extract_abstract_and_references(pdf_list):
    abstract_index = next((i for i, s in enumerate(pdf_list) if "ABSTRACT" or "Abstract"in s), None)
    references_index = next((i for i, s in enumerate(pdf_list) if "REFERENCES" in s), None)

    if abstract_index is not None and references_index is not None:
        abstract = pdf_list[abstract_index + 1:references_index]
        return abstract
    else:
        print("ABSTRACT or REFERENCES not found in the list.")
        return None

def convert_string_to_df(text):
    final_text=text.replace('\n',' ')
    final_text=final_text.split('.')
    final_text2=extract_abstract_and_references(final_text)
    return final_text2


def image_similarity(image_path1, image_path2):
    from skimage.metrics import structural_similarity as ssim
    from skimage import io,color
    # Load the images
    img1 = io.imread(image_path1)
    img2 = io.imread(image_path2)

    # Convert images to grayscale if they are in color
    if img1.shape[-1] == 3:
        img1 = color.rgb2gray(img1)
    if img2.shape[-1] == 3:
        img2 = color.rgb2gray(img2)

    # Compute the Structural Similarity Index (SSI)
    similarity_index, _ = ssim(img1, img2, full=True,data_range=img1.max() - img1.min())

    return similarity_index


# the below 2 functions check images for similarity and remove similar ones
def resize_image(image, target_size):
    resized_image = cv2.resize(image, target_size)
    return resized_image

import os
def compare_images(image1, image2):
    from skimage.metrics import structural_similarity as ssim
    img1=cv2.imread(image1)
    img2=cv2.imread(image2)
    # Resize images to a common size
    target_size = (width, height) = (300, 200)
    resized_image1 = resize_image(img1, target_size)
    resized_image2 = resize_image(img2, target_size)

    # Convert resized images to grayscale
    gray_image1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)

    # Compute Structural Similarity Index (SSI)
    similarity_index, _ = ssim(gray_image1, gray_image2, full=True)

    return similarity_index


# this function deletes the similar images
def remove_similar_images(image_names_arr, similarity_threshold=0.8):
    i = 0
    while i < len(image_names_arr):
        j = i + 1
        while j < len(image_names_arr):
            if compare_images(image_names_arr[i], image_names_arr[j]) > similarity_threshold:
                # Remove similar image and update the list
                os.remove(image_names_arr[j])
                image_names_arr.pop(j)
            else:
                j += 1
        
        i += 1

    return image_names_arr
import nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english') #stopwords


import spacy
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

import re
from nltk.tokenize import word_tokenize
def process_data(data_list):
    # renaming columns and lowering them
    data2=pd.Series(data_list)
    data2= data2.apply(lambda x: re.sub(r'[^a-zA-Z/s]+',' ',str(x)).lower())
    data_final=data2.apply(lambda x:" ".join([word for word in word_tokenize(x) if word not in stop_words and len(word)>2]))
    return data_final


# tokenizeing the above 
from gensim import corpora
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import gensim
def tokenize_data(dataframe): #this dataframe if from process_data
    word_list=[]
    for sentence in dataframe:
        word_list.extend(nltk.word_tokenize(sentence))   
    sent=[x.split() for x in dataframe]
    return sent

    # creating bigram and trigram
def create_bi_trigram(sentences):
    bigram=Phrases(sentences)
    trigram=Phrases(bigram[sentences])
    bigram_phraser=Phraser(bigram)
    trigram_phraser=Phraser(trigram)

    # creating bag of words
    bow=[trigram_phraser[bigram_phraser[words]] for words in sentences]

    return bow


# this function creates the corpus and dictionary(id2word) we need to pass to the model
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
def create_corpus(bow):
    bag_of_words = lemmatization(bow)
    id2word = corpora.Dictionary(bag_of_words)
    corpus = [id2word.doc2bow(sent) for sent in bag_of_words]
    return corpus,id2word


def clean_lda_output(lda_output):
    cleaned_output = []
    for topic in lda_output:
        cleaned_topic = []
        for word_prob in topic[1].split('+'):
            word = re.findall(r'"([^"]*)"', word_prob)[0]
            cleaned_topic.append(word)
        cleaned_output.append(cleaned_topic)
    return cleaned_output


import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import seaborn as sns
# this is where we initiate the model
def apply_LDA(corpus, id2word,num_topics):
    # Apply LDA model to the corpus and dictionary
    model = gensim.models.LdaModel(
        corpus=corpus,  # The corpus of documents
        id2word=id2word,  # The dictionary mapping word IDs to words
        num_topics=num_topics  # Number of topics
    )
    
    topics=clean_lda_output(model.print_topics())
    return model
    # Print the topics learned by the LDA model
        







from gensim.models import CoherenceModel
def calculate_coherence(model, bag_of_words, id2word):

    # Create a CoherenceModel object for the given LDA model
    coherence_model_lda = CoherenceModel(
        model=model,  # The LDA model
        texts=bag_of_words,  # The bag of words
        dictionary=id2word,  # The dictionary mapping word IDs to words
        coherence='c_v'  # The coherence model to use
    )

    # Calculate the coherence score
    coherence_lda = coherence_model_lda.get_coherence()

    # Print the coherence score
    # print('\nCoherence Score: ', coherence_lda)
    return coherence_lda



# below is the order in which the functions are to be executed

# 1)process_pdf
# 2)convert_string_to_df
# 4)remove_similar_images
# 5)process_data
# 6)tokenize_data
# 7)create_bi_trigram
# 8)create_corpus
# 9)apply_LDA
# 10)calculate_coherence
    
    # example:


# pdf_text=process_pdf(path)
# dataframe=convert_string_to_df(pdf_text)
# remove_similar_images(image_names_arr)
# new_dataframe=process_data(dataframe)
# sentences=tokenize_data(new_dataframe)
# bag_of_words=create_bi_trigram(sentences)
# corpus,id2word=create_corpus(bag_of_words)
# apply_LDA(corpus, id2word)

