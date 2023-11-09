import streamlit as st

# Title and Sidebar Title
st.title("2023 Developer Salary Web App")
st.sidebar.title("2023 Developer Salary Web App")

# Radio button for user selection
app_mode = st.sidebar.radio("Select App Mode", ["Salary Analysis", "Predict Salary"])

# Display content based on user selection
if app_mode == "Salary Analysis":
    st.markdown("You've selected Salary Analysis.")
    # Add code for salary analysis here
elif app_mode == "Predict Salary":
    st.markdown("You've selected Predict Salary.")
    # Add code for salary prediction here