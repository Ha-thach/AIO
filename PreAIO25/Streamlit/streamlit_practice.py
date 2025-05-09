import streamlit as st
# st.title("This is my first Streamlit app")
# st.text("Streamlit is so easy")
# st.header("This is a header")
#
# st.header("This is a header with a divider", divider=True);
# st.header(":blue[This is a header with a divider]", divider=True);
# st.subheader("Example of subheader", divider=True);

#st.markdown: to print a document

# document = """
# There are some basic rules to format text in Streamlit using markdown:
#
# ## Heading: Use hash signs before the text (From 1 to 6 # hashes)
# *Italic text*: Enclose with one asterisk on each side  *
# **Bold text**: Enclose with two asterisks on each side
# ***Bold and Italic***: Enclose with three asterisks on each side
#
# - Bullet points: Use a dash (-) or asterisk (*) at the beginning of the line
# 1. Numbered lists: Use numbers followed by a period
#
# > Blockquotes: Use a greater-than symbol (>) before the text
#
# `Inline code`: Enclose with backticks (\`)
# ```python
# # Code block: Use triple backticks and optionally specify the language
# def hello():
#     print("Hello, Streamlit!")
# """
# st.markdown(document)

#st.text_input
# st.title("This is my first Streamlit app!")
# name =st.text_input(label="Enter your name: ",value="",)
# age = st.number_input("Enter your age: ", value =0, step=1, format ="%d") #spinner: increase and decrease by -+
# grade = st.slider("Which grade are you in", 0, 1, 12) #slider
# st.write("Your name is: ", name)
# st.write("Your age is: ", age)
# st.write("Your grade is: ", grade)
# is_aio = st.checkbox("Are you AIO?")
# if is_aio:
#     st.checkbox("Do you finish the contest?")
#
# option = st.selectbox(
#     "Which OS do you use to learn Deep Learning?", ("Windows", "Ubuntu", "MacOS"),
#     index=None, placeholder="Choose an option",
# )
# st.write("You chose:", option)
#
# option1 = st.radio(
#     "Which OS do you use to learn Deep Learning?",["Windows", "Ubuntu", "MacOS"],
#     captions=[
#         "Windows is good to play games :video_game:",
#         "Ubuntu is good for learning CLI :smile:",
#         "MacOS is good for designer :vampire:"
#     ]
# ) #looks like Multiple choice
# st.write("You chose:", option1)

#Example of using button

# st.title("This is my first Streamlit app!")
# email= st.text_input("Type your email here:")
# if st.button("Submit"):
#     if "@" not in email:
#         st.write("Please type a valid email!")
#     else:
#         st.write("Submitted. Your email is: ", email)

#Example of using file uploader
from PIL import Image

uploaded_file=st.file_uploader("Choose an image!", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img =Image.open(uploaded_file)
    img = img.resize((224,224))
    st.image(img, caption="Uploaded Image.")


st.write("I love AI VIET NAM")

st.write("# I love AI VIET NAM")

st.write("## I love AI VIET NAM")

st.write("### I love AI VIET NAM")

st.write("#### I love AI VIET NAM")

st.write("##### I love AI VIET NAM")

st.write("###### I love AI VIET NAM")
st.write(" I love AI VIET NAM")i