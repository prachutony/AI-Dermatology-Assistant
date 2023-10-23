import streamlit as st
import pymongo
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu
from contextlib import contextmanager

st.set_page_config(initial_sidebar_state="expanded")

client = pymongo.MongoClient(**st.secrets["mongo"])
db = client['skin_detector']
collection = db["passwords"]

def get_doctors():
    # Find documents where "doctor" is true
    cursor = collection.find({"doctor": True})

    # Extract usernames from the matching documents
    doctor_usernames = [doc["username"] for doc in cursor]

    return doctor_usernames



def show_profile(username):
    st.title(username)
    st.markdown('<h3>Description:</h3>', unsafe_allow_html=True)
    description = get_description(username)
    
    if description:
        st.markdown(f'<ul>', unsafe_allow_html=True)
        for line in description:
            st.markdown(f'<li>{line}</li>', unsafe_allow_html=True)
        st.markdown(f'</ul>', unsafe_allow_html=True)
    else:
        st.write("Description is not uploaded")



def get_description(username):
    query = {"username": username}
    result = collection.find_one(query)
    description_list = result.get('description', [])
    return description_list



def update_description(username, description):
    filter = {'username': username}
    update = {'$set': {'description': description}}
    collection.update_one(filter, update)



def update_username(old_username, new_username):
    filter = {'username': old_username}
    update = {'$set': {'username': new_username}}
    collection.update_one(filter, update)
    return new_username



def update_email(username, email):
    filter = {'username': username}
    update = {'$set': {'email': email}}
    collection.update_one(filter, update)



def update_password(username, password):
    filter = {'username': username}
    update = {'$set': {'password': password}}
    collection.update_one(filter, update)



def add_user(username, email, password, option):
    data = {
        'username': username,
        'email': email,
        'password': password,
        'doctor': option
    }
    collection.insert_one(data)


def check_username(username):
    usernames = [user['username'] for user in collection.find({})]
    return username not in usernames



def check_email(email):
    emails = [mail['email'] for mail in collection.find({})]
    return email not in emails



def validate_user(username, password):
    query = {"username": username, "password": password}
    return collection.find_one(query) is not None


def update_username_section():
    username_updater = st.text_input("Edit your username")
    if st.button("Update Username"):
        if username_updater:
            if check_username(username_updater):
                st.session_state.username = update_username(st.session_state.username, username_updater)
                st.success("Username updated successfully")
            else:
                st.error("Username already exists")
        else:
            st.error("Username should not be empty")

def update_email_section():
    email_updater = st.text_input("Edit your Email")
    if st.button("Update Email"):
        if email_updater:
            if check_email(email_updater):
                update_email(st.session_state.username, email_updater)
                st.success("Email updated successfully")
            else:
                st.error("Email already linked with another account")
        else:
            st.error("Email should not be empty")

def update_password_section():
    password_updater = st.text_input("Change your Password", type="password")
    if st.button("Change Password"):
        if password_updater:
            update_password(st.session_state.username, password_updater)
            st.success("Password updated successfully")
        else:
            st.error("Password should not be empty")

def update_description_section():
    discription = st.text_area("Describe yourself")
    if st.button("Update Description"):
        lines = discription.split('\n')
        update_description(st.session_state.username, lines)
        st.success("Description updated successfully")



def login():
    if st.session_state.login == False:
        loginholder = st.empty()
        loginholder_exit = False
        with loginholder.container():
            select = st.selectbox('Login / Sign Up', ['Login', 'Sign Up'])
        
            if select == "Sign Up":
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                doctor_or_patient = st.selectbox("Doctor/Patient", ['Doctor', 'Patient'])

                if st.button("Sign Up"):
                    if username and email and password:
                        if check_username(username) and check_email(email):
                            add_user(username, email, password, doctor_or_patient == "Doctor")
                            st.success("Account created Successfully. Please log in with the credentials submitted!")
                        else:
                            st.error("Username or Email already exists. Try logging in.")
                    else:
                        st.error("Invalid credentials. Please try again.")

            elif select == "Login":
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.button("Login"):
                    if username and password:
                        if validate_user(username, password):
                            loginholder_exit = True
                            st.session_state.username = username
                            st.session_state.login = True
                        else:
                            st.error("Invalid Username or password")
                    else:
                        st.error("Please fill every box")
        if loginholder_exit == True:
            loginholder.empty()

def predictor(model):

    st.title("Skin Infection Detector")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Perform inference when an image is uploaded
        image = tf.image.decode_image(uploaded_image.read(), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (160, 160))
        image = np.expand_dims(image, axis=0)

        # Make predictions using your loaded model
        predictions = model.predict(image)

        # Find the index of the class with the highest probability
        highest_prob_index = np.argmax(predictions)

        # Display the results
        st.image(uploaded_image, caption="Your uploaded image", use_column_width=True)
        if highest_prob_index == 0:
            #st.write("Prediction:", predictions)
            st.success("This looks like cellulitis")
        elif highest_prob_index == 1:
            #st.write("Prediction:", predictions)
            st.success("This looks like skin abscess")
        elif highest_prob_index == 2:
            #st.write("Prediction:", predictions)
            st.success("This looks like candida")
        elif highest_prob_index == 3:
            #st.write("Prediction:", predictions)
            st.success("This looks like purpura")
        elif highest_prob_index == 4:
            #st.write("Prediction:", predictions)
            st.success("This looks like hematoma")
        else:
            st.error("Sorry, the image is not clear")
        

def edit_profile():
    st.title("Edit your profile")
    update_username_section()
    update_email_section()
    update_password_section()
    update_description_section()

def chat_page():
    if st.session_state.chat == False:
        st.title("Chat with a doctor")
        doctors = get_doctors()
        doctors.sort()
        if st.session_state.username in doctors:
            doctors.remove(st.session_state.username)
        selected_doctor = st.selectbox("Select a doctor to chat" , doctors)
            
        if selected_doctor:
            show_profile(selected_doctor)
            if st.button(f"Chat with {selected_doctor}"):
                st.session_state.doctor = selected_doctor
                st.session_state.chat = True


def chat_box():
    # st.warning("To go back please Double click on the go back button")
    # if st.button("Go back"):
    #     st.session_state.chat == False
        

    st.title(f"Chat with {st.session_state.doctor}")
    chat_collections = db.list_collection_names()
    chat_collection = ""
    for name in chat_collections:
        if st.session_state.username in name and st.session_state.doctor in name:
            chat_collection = db[name]

    if chat_collection == "":
        chat_collection = db[f"{st.session_state.username}{st.session_state.doctor}chat"]
        # Create the collection if it doesn't exist

    # Initialize chat history
    chat_cursor = chat_collection.find({})

    # Display chat messages from the database on app rerun
    for message in chat_cursor:
        with st.chat_message(message["role"]):
            username = message["username"]
            content = message["content"]
            if username == st.session_state.username:
                st.markdown(f"YOU : {content}")
            else:
                st.markdown(f"{username} : {content}")

    # React to user input
    if prompt := st.chat_input("What is up?", key="chat_2"):
        # Display user message in chat message container
        st.chat_message("user").markdown(f"YOU : {prompt}")

        # Add user message to chat history in MongoDB
        user_message = {"role": "user", "username" : st.session_state.username,"content": prompt}
        chat_collection.insert_one(user_message)


@st.cache_resource
def load_model():
    # Load the .h5 model
    model_path = 'Vgg_model.h5'
    model = tf.keras.models.load_model(model_path)
    return model


def main():
    if not hasattr(st.session_state, 'login'):
        st.session_state.login = False
    
    if not hasattr(st.session_state, 'username'):
        st.session_state.username = ""

    if not hasattr(st.session_state, 'chat'):
        st.session_state.chat = False

    if not hasattr(st.session_state, 'doctor'):
        st.session_state.doctor = ""

    login()

    if st.session_state.login == True:
        model = load_model()

        

        st.sidebar.title(f"Welcome {st.session_state.username} !")

        with st.sidebar:
            navigated = option_menu(
                menu_title=None,
                options=["Predictor", "View your profile", "Edit your profile", "Chat with a Doctor"]
            )
        
        
        if navigated == "Predictor":
            predictor(model)
        
        elif navigated == "View your profile":
            show_profile(st.session_state.username)

        elif navigated == "Edit your profile":
            edit_profile()
        elif navigated == "Chat with a Doctor":
            chatholder = st.empty()
            with chatholder.container():
                chat_page()

            if st.session_state.chat == True:
                chatholder.empty()
                
                if st.button("Go back"):
                    st.session_state.chat = False
                    st.warning("Please click again to go back")
                chat_box()


        if st.sidebar.button("Logout"):
            st.session_state.login = False
            st.session_state.chat = False
            st.sidebar.warning("Please click again to logout")
        



if __name__ == "__main__":
    main()

