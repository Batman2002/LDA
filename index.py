import streamlit as st
from time import sleep
from db_trial import connect_to_database, verify_login,register_user,get_connection

print("Trying to connect")
mydb=get_connection()


# make_sidebar()
st.set_page_config(initial_sidebar_state="collapsed")


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


username = st.text_input("Username")
password = st.text_input("Password", type="password")

st.session_state.username = username

if st.button("Register"):
    # new_username = st.text_input("New Username")
    # new_password = st.text_input("New Password", type="password")
    register_user(username, password)
    st.success("User registered successfully!")
    st.switch_page("pages/front.py")


if st.button("Log in", type="primary"):
    if verify_login(username,password): 
        st.session_state.logged_in = True
        st.success("Logged in successfully!")
        sleep(0.5)
        st.switch_page("pages/front.py")
    else:
        st.error("Incorrect username or password")
