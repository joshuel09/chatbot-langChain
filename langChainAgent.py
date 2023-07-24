#pip install streamlit langchain openai faiss-cpu tiktoken
import os
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
# from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
#
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory




user_api_key = st.sidebar.text_input(
    label="OpenAI API key",
    placeholder="Paste your openAI API key, sk-",
    type="password")

uploaded_file = st.sidebar.file_uploader("upload", type=['png', 'jpg'])


# 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
os.environ['OPENAI_API_KEY'] = user_api_key

if uploaded_file :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
    # data = loader.load()

    # loader = PyPDFLoader(file_path=tmp_file_path)  
    # data = loader.load_and_split(text_splitter)

    # embeddings = OpenAIEmbeddings()
    # vectors = FAISS.from_documents(data, embeddings)

    class ImageCaptionTool(BaseTool):
        name = "Image captioner"
        description = "Use this tool when given the path to an image that you would like to be described. " \
                    "It will return a simple caption describing the image."

        def _run(self, img_path):
            image = Image.open(img_path).convert('RGB')

            model_name = "Salesforce/blip-image-captioning-large"
            device = "cpu"  # cuda

            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

            inputs = processor(image, return_tensors='pt').to(device)
            output = model.generate(**inputs, max_new_tokens=20)

            caption = processor.decode(output[0], skip_special_tokens=True)

            return caption

        def _arun(self, query: str):
            raise NotImplementedError("This tool does not support async")


    class ObjectDetectionTool(BaseTool):
        name = "Object detector"
        description = "Use this tool when given the path to an image that you would like to detect objects. " \
                    "It will return a list of all detected objects. Each element in the list in the format: " \
                    "[x1, y1, x2, y2] class_name confidence_score."

        def _run(self, img_path):
            image = Image.open(img_path).convert('RGB')

            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # convert outputs (bounding boxes and class logits) to COCO API
            # let's only keep detections with score > 0.9
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            detections = ""
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                detections += ' {}'.format(model.config.id2label[int(label)])
                detections += ' {}\n'.format(float(score))

            return detections

        def _arun(self, query: str):
            raise NotImplementedError("This tool does not support async")
        
    def get_image_caption(image_path):
        """
        Generates a short caption for the provided image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: A string representing the caption for the image.
        """
        image = Image.open(image_path).convert('RGB')

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"  # cuda

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption


    def detect_objects(image_path):
        """
        Detects objects in the provided image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: A string with all the detected objects. Each object as '[x1, x2, y1, y2, class_name, confindence_score]'.
        """
        image = Image.open(image_path).convert('RGB')

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))

        return detections


    #initialize the agent
    tools = [ImageCaptionTool(), ObjectDetectionTool()]

    llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo')

    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    agent_chain = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        max_iterations=5,
        verbose=True,
        memory=conversational_memory,
        early_stopping_method='generate'
    )


    # chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo-16k'),
    #                                                                   retriever=vectors.as_retriever())


    def conversational_chat(query):
        
        result = agent_chain.run(f'{query}, this is the image path: {tmp_file_path}')
        st.session_state['history'].append((query, result))
        
        return result
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Feel free to ask about anything regarding this" + uploaded_file.name]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey !"]
        
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk about your pdf data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
                
#streamlit run tuto_chatbot_csv.py