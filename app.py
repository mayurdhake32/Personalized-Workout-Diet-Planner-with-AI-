import os
from datetime import datetime, date

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
- Activity level: {activity_level}
- Estimated daily calorie target: {calorie_target} kcal
- Other instructions: {comments}

Return the plans in a neat, structured format using Markdown tables.

Include relevant key notes, and try to keep the diet plan roughly aligned with
the estimated daily calorie target provided above.

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
        "activity_level",
        "calorie_target",
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
st.write("Create a personalized workout and diet plan, track your progress, and chat with your plan — all in one place.")


# --------------------------------------------------
# 7. Session State Initialization
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "plan" not in st.session_state:
    st.session_state.plan = None

if "progress_log" not in st.session_state:
    st.session_state.progress_log = []  # list of {"date": date, "weight": float}

if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {}


# --------------------------------------------------
# 8. Health Calculation Helpers
# --------------------------------------------------

ACTIVITY_MULTIPLIERS = {
    "Sedentary (little/no exercise)": 1.2,
    "Lightly active (1-3 days/week)": 1.375,
    "Moderately active (3-5 days/week)": 1.55,
    "Very active (6-7 days/week)": 1.725,
    "Extremely active (athlete)": 1.9,
}


def calculate_bmi(weight_kg, height_cm):
    if not height_cm:
        return None
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)


def bmi_category(bmi):
    if bmi is None:
        return "Unknown"
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def calculate_bmr(weight_kg, height_cm, age, gender):
    # Mifflin-St Jeor Equation
    if gender == "Male":
        return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    elif gender == "Female":
        return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    else:
        return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 78


def calculate_tdee(bmr, activity_level):
    return bmr * ACTIVITY_MULTIPLIERS.get(activity_level, 1.2)


def calculate_calorie_target(tdee, current_weight, target_weight):
    if target_weight < current_weight:
        return round(tdee - 500)  # ~0.5kg/week loss
    elif target_weight > current_weight:
        return round(tdee + 300)  # gradual surplus
    else:
        return round(tdee)


def estimate_weeks_to_goal(current_weight, target_weight):
    diff = abs(current_weight - target_weight)
    if diff == 0:
        return 0
    return round(diff / 0.5)  # ~0.5 kg per week


# --------------------------------------------------
# 9. Tabbed Navigation
# --------------------------------------------------

tab_plan, tab_dashboard, tab_progress, tab_chat = st.tabs(
    ["📝 Generate Plan", "📊 Dashboard", "📈 Progress Tracker", "💬 Chat With Plan"]
)


# ====================================================
# TAB 1: PLAN GENERATOR
# ====================================================

with tab_plan:

    col1, col2 = st.columns(2)

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

        height_cm = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=250.0,
            value=170.0,
            step=1.0
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

        activity_level = st.selectbox(
            "Activity Level",
            list(ACTIVITY_MULTIPLIERS.keys()),
            index=1
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

        # Live calculated stats shown before generating
        bmi_live = calculate_bmi(current_weight, height_cm)
        bmr_live = calculate_bmr(current_weight, height_cm, age, gender)
        tdee_live = calculate_tdee(bmr_live, activity_level)
        calorie_target_live = calculate_calorie_target(tdee_live, current_weight, target_weight)

        st.info(
            f"**Quick stats:** BMI {bmi_live} ({bmi_category(bmi_live)}) · "
            f"BMR {round(bmr_live)} kcal · TDEE {round(tdee_live)} kcal · "
            f"Suggested daily target {calorie_target_live} kcal"
        )

        if st.button("🚀 Generate Plans", use_container_width=True):

            # Clear previous chat history
            st.session_state.messages = []

            # Save inputs for dashboard use
            st.session_state.user_inputs = {
                "height_cm": height_cm,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "age": age,
                "gender": gender,
                "activity_level": activity_level,
                "number_of_weeks": number_of_weeks,
            }

            # Seed progress log with the starting weight if empty
            if not st.session_state.progress_log:
                st.session_state.progress_log.append(
                    {"date": date.today(), "weight": current_weight}
                )

            with st.spinner("Generating your personalized fitness and diet plan..."):

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
                            "activity_level": activity_level,
                            "calorie_target": calorie_target_live,
                            "number_of_weeks": number_of_weeks,
                            "comments": comments,
                        }
                    )

                    # Store only the text content
                    st.session_state.plan = response.content

                    st.success("✅ Plans generated successfully! Check the Dashboard and Chat tabs.")

                except Exception as e:

                    st.error(f"An error occurred: {e}")

    with col2:

        if st.session_state.plan:

            st.header("📋 Your Personalized Plan")

            st.markdown(st.session_state.plan)

            st.download_button(
                label="⬇️ Download Plan (Markdown)",
                data=st.session_state.plan,
                file_name="fitness_diet_plan.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.info("Fill in your details and click **Generate Plans** to see your plan here.")


# ====================================================
# TAB 2: INTERACTIVE DASHBOARD
# ====================================================

with tab_dashboard:

    st.header("📊 Your Health Dashboard")

    if not st.session_state.user_inputs:
        st.info("Generate a plan first (in the 'Generate Plan' tab) to populate your dashboard.")
    else:

        ui = st.session_state.user_inputs
        bmi = calculate_bmi(ui["current_weight"], ui["height_cm"])
        bmr = calculate_bmr(ui["current_weight"], ui["height_cm"], ui["age"], ui["gender"])
        tdee = calculate_tdee(bmr, ui["activity_level"])
        calorie_target = calculate_calorie_target(tdee, ui["current_weight"], ui["target_weight"])
        weeks_needed = estimate_weeks_to_goal(ui["current_weight"], ui["target_weight"])
        water_target_l = round(ui["current_weight"] * 0.033, 1)

        # ---- KPI Row ----
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("BMI", f"{bmi}", bmi_category(bmi))
        k2.metric("BMR", f"{round(bmr)} kcal")
        k3.metric("TDEE", f"{round(tdee)} kcal")
        k4.metric("Daily Calorie Target", f"{calorie_target} kcal")
        k5.metric("Water Target", f"{water_target_l} L/day")

        st.markdown("---")

        d1, d2 = st.columns(2)

        with d1:
            # ---- BMI Gauge ----
            fig_bmi = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bmi,
                title={"text": "BMI"},
                gauge={
                    "axis": {"range": [10, 40]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [10, 18.5], "color": "#a8d0f0"},
                        {"range": [18.5, 25], "color": "#a8f0b0"},
                        {"range": [25, 30], "color": "#f0e0a8"},
                        {"range": [30, 40], "color": "#f0a8a8"},
                    ],
                }
            ))
            fig_bmi.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_bmi, use_container_width=True)

        with d2:
            # ---- Weight Projection Chart ----
            direction = -0.5 if ui["target_weight"] < ui["current_weight"] else 0.5
            weeks_range = list(range(0, max(weeks_needed, ui["number_of_weeks"]) + 1))
            projected_weights = [
                round(ui["current_weight"] + direction * w, 1) for w in weeks_range
            ]
            # Clip projection so it doesn't overshoot the target
            if direction < 0:
                projected_weights = [max(w, ui["target_weight"]) for w in projected_weights]
            else:
                projected_weights = [min(w, ui["target_weight"]) for w in projected_weights]

            df_proj = pd.DataFrame({"Week": weeks_range, "Projected Weight (kg)": projected_weights})
            fig_proj = px.line(
                df_proj, x="Week", y="Projected Weight (kg)", markers=True,
                title="Projected Weight Trajectory"
            )
            fig_proj.add_hline(
                y=ui["target_weight"], line_dash="dot", line_color="green",
                annotation_text="Target", annotation_position="bottom right"
            )
            fig_proj.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_proj, use_container_width=True)

        d3, d4 = st.columns(2)

        with d3:
            # ---- Calories: intake vs burn ----
            fig_cal = go.Figure(data=[
                go.Bar(name="TDEE (Maintenance)", x=["Calories"], y=[round(tdee)], marker_color="#f0a8a8"),
                go.Bar(name="Daily Target", x=["Calories"], y=[calorie_target], marker_color="#a8f0b0"),
            ])
            fig_cal.update_layout(
                barmode="group", title="Maintenance vs. Target Calories",
                height=320, margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_cal, use_container_width=True)

        with d4:
            # ---- Estimated macro split (simple general-purpose split) ----
            protein_pct, carbs_pct, fat_pct = 30, 40, 30
            fig_macro = px.pie(
                names=["Protein", "Carbohydrates", "Fat"],
                values=[protein_pct, carbs_pct, fat_pct],
                title="Suggested Macro Split (%)",
                hole=0.4,
            )
            fig_macro.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_macro, use_container_width=True)

        st.caption(
            f"Estimated time to reach your target weight at a safe pace (~0.5 kg/week): "
            f"**{weeks_needed} week(s)**."
        )


# ====================================================
# TAB 3: PROGRESS TRACKER
# ====================================================

with tab_progress:

    st.header("📈 Weight Progress Tracker")
    st.write("Log your weight regularly to visualize your progress toward your goal.")

    with st.form("progress_form", clear_on_submit=True):
        pcol1, pcol2, pcol3 = st.columns([2, 2, 1])
        log_date = pcol1.date_input("Date", value=date.today())
        log_weight = pcol2.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.1)
        submitted = pcol3.form_submit_button("➕ Add Entry", use_container_width=True)

        if submitted:
            st.session_state.progress_log.append({"date": log_date, "weight": log_weight})
            st.success(f"Logged {log_weight} kg on {log_date}.")

    if st.session_state.progress_log:

        df_log = pd.DataFrame(st.session_state.progress_log).sort_values("date")
        df_log["date"] = pd.to_datetime(df_log["date"])

        fig_progress = px.line(
            df_log, x="date", y="weight", markers=True,
            title="Weight Over Time", labels={"date": "Date", "weight": "Weight (kg)"}
        )

        if st.session_state.user_inputs:
            fig_progress.add_hline(
                y=st.session_state.user_inputs["target_weight"],
                line_dash="dot", line_color="green",
                annotation_text="Target", annotation_position="bottom right"
            )

        st.plotly_chart(fig_progress, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(df_log.rename(columns={"date": "Date", "weight": "Weight (kg)"}), use_container_width=True, hide_index=True)
        with c2:
            csv_data = df_log.rename(columns={"date": "Date", "weight": "Weight (kg)"}).to_csv(index=False)
            st.download_button(
                "⬇️ Download Progress Log (CSV)",
                data=csv_data,
                file_name="weight_progress_log.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if st.button("🗑️ Clear Progress Log", use_container_width=True):
                st.session_state.progress_log = []
                st.rerun()
    else:
        st.info("No entries yet. Add your first weight log above.")


# ====================================================
# TAB 4: CHAT WITH PLAN
# ====================================================

with tab_chat:

    st.header("💬 Chat With Your Plan")

    if not st.session_state.plan:
        st.info("Generate a plan first (in the 'Generate Plan' tab) before chatting.")
    else:

        # Display previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        user_question = st.chat_input("Ask a question about your plan...")

        if user_question:

            # Display user question
            st.session_state.messages.append({"role": "user", "content": user_question})

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
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Display AI response
            with st.chat_message("assistant"):
                st.write(answer)


# --------------------------------------------------
# 10. Footer
# --------------------------------------------------

st.markdown("---")
st.caption("Created by MAYUR MAHESH DHAKE")
