import os

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


# --------------------------------------------------
# 1. Load API Key
# --------------------------------------------------

load_dotenv("key.env")

api_key = os.getenv("GROQ_API_KEY")

# For Streamlit Cloud
if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()


# --------------------------------------------------
# 2. Initialize Groq LLM
# --------------------------------------------------

langchain_llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.7
)


# --------------------------------------------------
# 3. Plan Prompt
# --------------------------------------------------

plan_prompt_template = """
You are a fitness and diet planner.

Using the following user inputs, create two detailed plans:

1. A diet plan table listing day-to-day food intake for {number_of_weeks} weeks.
2. A workout plan table listing day-to-day exercises for {number_of_weeks} weeks.

User Inputs:

- Workout type: {workout_type}
- Diet type: {diet_type}
- Current body weight: {current_weight} kg
- Target weight: {target_weight} kg
- Dietary restrictions: {dietary_restrictions}
- Health conditions: {health_conditions}
- Age: {age}
- Gender: {gender}
- Other instructions: {comments}

Return the plans in a neat, structured format using Markdown tables.

Include relevant key notes.

Important:
If the user has a health condition, include a recommendation to consult a qualified healthcare professional before following a new diet or workout plan.
"""

plan_prompt = PromptTemplate(
    input_variables=[
        "workout_type",
        "diet_type",
        "current_weight",
        "target_weight",
        "dietary_restrictions",
        "health_conditions",
        "age",
        "gender",
        "number_of_weeks",
        "comments",
    ],
    template=plan_prompt_template,
)


# --------------------------------------------------
# 4. Chat Prompt
# --------------------------------------------------

chat_prompt_template = """
You are a fitness and diet assistant.

Answer the user's question based on the following personalized plan.

PLAN:
{plan}

USER QUESTION:
{question}

Provide a clear and helpful answer.

If the question involves a medical condition, injury, or serious health concern,
recommend consulting a qualified healthcare professional.
"""

chat_prompt = PromptTemplate(
    input_variables=["plan", "question"],
    template=chat_prompt_template
)


# --------------------------------------------------
# 5. Create Modern LangChain Chains
# --------------------------------------------------

plan_chain = plan_prompt | langchain_llm

chat_chain = chat_prompt | langchain_llm


# --------------------------------------------------
# 6. Streamlit Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="🏋️ Fitness and Diet Planner",
    page_icon="🏋️",
    layout="wide"
)

st.title("🏋️ Fitness and Diet Planner")
st.write("Create a personalized workout and diet plan using AI.")


# --------------------------------------------------
# 7. Two-Column Layout
# --------------------------------------------------

col1, col2 = st.columns(2)


# --------------------------------------------------
# 8. User Input Section
# --------------------------------------------------

with col1:

    st.header("Enter Your Details")

    workout_type = st.text_input(
        "Workout Type",
        placeholder="Example: Weight Loss, Muscle Gain"
    )

    diet_type = st.text_input(
        "Diet Type",
        placeholder="Example: Indian, Mediterranean"
    )

    current_weight = st.number_input(
        "Current Body Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=75.0,
        step=1.0
    )

    target_weight = st.number_input(
        "Target Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=68.0,
        step=1.0
    )

    dietary_restrictions = st.text_input(
        "Dietary Restrictions",
        placeholder="Example: No dairy, Low sugar"
    )

    health_conditions = st.text_input(
        "Any Health Conditions?",
        placeholder="Example: Diabetes, Knee pain, None"
    )

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=30,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    number_of_weeks = st.slider(
        "Number of Weeks",
        min_value=1,
        max_value=12,
        value=4
    )

    comments = st.text_area(
        "Additional Comments",
        placeholder="Enter any additional instructions..."
    )


    # --------------------------------------------------
    # Generate Plan Button
    # --------------------------------------------------

    if st.button(
        "🚀 Generate Plans",
        use_container_width=True
    ):

        # Clear previous chat history
        st.session_state.messages = []

        with st.spinner(
            "Generating your personalized fitness and diet plan..."
        ):

            try:

                response = plan_chain.invoke(
                    {
                        "workout_type": workout_type,
                        "diet_type": diet_type,
                        "current_weight": current_weight,
                        "target_weight": target_weight,
                        "dietary_restrictions": dietary_restrictions,
                        "health_conditions": health_conditions,
                        "age": age,
                        "gender": gender,
                        "number_of_weeks": number_of_weeks,
                        "comments": comments,
                    }
                )

                # Store only the text content
                st.session_state.plan = response.content

                st.success(
                    "✅ Plans generated successfully!"
                )

            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )


# --------------------------------------------------
# 9. Display Generated Plan
# --------------------------------------------------

with col2:

    if (
        "plan" in st.session_state
        and st.session_state.plan
    ):

        st.header("📋 Your Personalized Plan")

        st.markdown(
            st.session_state.plan
        )


# --------------------------------------------------
# 10. Chat Section
# --------------------------------------------------

if (
    "plan" in st.session_state
    and st.session_state.plan
):

    st.markdown("---")

    st.subheader("💬 Chat With Your Plan")


    # Initialize chat history
    if "messages" not in st.session_state:

        st.session_state.messages = []


    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])


    # Chat input
    user_question = st.chat_input(
        "Ask a question about your plan..."
    )


    if user_question:

        # Display user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )


        with st.chat_message("user"):

            st.write(user_question)


        # Get AI answer
        try:

            response = chat_chain.invoke(
                {
                    "plan": st.session_state.plan,
                    "question": user_question
                }
            )

            answer = response.content


        except Exception as e:

            answer = f"An error occurred: {e}"


        # Store AI response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        # Display AI response
        with st.chat_message("assistant"):

            st.write(answer)


# --------------------------------------------------
# 11. Footer
# --------------------------------------------------

st.markdown("---")

st.caption("Created by Vikram Bhat")
