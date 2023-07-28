import os
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
import tempfile
from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from deepface import DeepFace
import cv2 as cv
import time


user_api_key = st.sidebar.text_input(
    label="OpenAI API key",
    placeholder="Paste your openAI API key, sk-",
    type="password")

uploaded_file = st.sidebar.file_uploader("upload", type=['png', 'jpg'])

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

    def getFaceBox(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)    
        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
            return frameOpencvDnn, bboxes

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
            output = model.generate(**inputs, max_new_tokens=4000)

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
        
    class EmotionDetectionTool(BaseTool):
        name = "Emotion detector"
        description = "Use this tool when given the path to an image that you would like to detect emotion. " 

        def _run(self, img_path):

            detections = DeepFace.analyze(img_path)

            return detections

        def _arun(self, query: str):
            raise NotImplementedError("This tool does not support async")
        
    class GenderAgeDetectionTool(BaseTool):
        
        name = "Gender and age detector"
        description = "Use this tool when given the path to an image that you would like to detect Gender and Age. " 

        def _run(self, img_path):

            padding = 20

            t = time.time()
            frame = cv.imread(img_path)
            frameFace, bboxes = getFaceBox(faceNet, frame)
            for bbox in bboxes:
                # print(bbox)
                face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                # print("Gender Output : {}".format(genderPreds))
                print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                print("Age Output : {}".format(agePreds))
                print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
                label = "{},{}".format(gender, age)
                # cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
                return {"Gender":gender, "Age":age}

        def _arun(self, query: str):
            raise NotImplementedError("This tool does not support async")
        
    
    faceProto = "AgeGender/opencv_face_detector.pbtxt"
    faceModel = "AgeGender/opencv_face_detector_uint8.pb"
    ageProto = "AgeGender/age_deploy.prototxt"
    ageModel = "AgeGender/age_net.caffemodel"
    genderProto = "AgeGender/gender_deploy.prototxt"
    genderModel = "AgeGender/gender_net.caffemodel"

    # Load network
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(34-39)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female'] 


    #initialize the agent
    tools = [ImageCaptionTool(), ObjectDetectionTool(), EmotionDetectionTool(), GenderAgeDetectionTool()]

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