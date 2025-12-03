import json
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from groq import Groq
from sklearn.metrics import f1_score, accuracy_score

try:
    groq_client = Groq()
except Exception as e:
    st.error(f"Error initializing Groq client: {e}. Please ensure the GROQ_API_KEY environment variable is set.")
    groq_client = None

# =================================================================
# L-PAST CORE FUNCTIONS
# =================================================================

def calculate_p_ai(S_score_humanity, T_variance_iat):
    """Calculates the Probability of AI based on Semantic and Timing scores."""

    WEIGHT_S = 0.48
    WEIGHT_T = 0.52

    MAX_EXPECTED_VARIANCE = 500000.0
    MAX_LLM_SCORE = 100.0

    S_Normalized = 1.0 - (S_score_humanity / MAX_LLM_SCORE)
    T_variance_clipped = np.clip(T_variance_iat, 0.0, MAX_EXPECTED_VARIANCE)
    T_Normalized = 1.0 - (T_variance_clipped / MAX_EXPECTED_VARIANCE)

    P_AI = (WEIGHT_S * S_Normalized) + (WEIGHT_T * T_Normalized)
    return P_AI * 100.0


def get_attribution(p_ai):
    if p_ai >= 95.0:
        return 'AI-Automated'
    elif p_ai >=70.0:
        return 'Machine-Assisted'
    else:
        return 'Human-Led'

# semantic score function
def get_live_s_score(command_text: str) -> int:
    """
    Queries Groq with the L-PAST prompt and extracts the semantic score.
    """
    if groq_client is None:
        return 50

    clean_command_data = json.dumps({"command_data": command_text})

    # --- L-PAST Prompt Definition ---
    LPAST_PROMPT = f"""
    You are an expert semantic attribution analyst. Your task is to score the provided command text from 0 to 100.
    Score 100 means the text is PURELY HUMAN (high noise/errors).
    Score 0 means the text is PURELY AUTOMATED (high precision/algorithmic).

    CRITICAL INSTRUCTION: **You MUST use the full range of integers from 0 to 100.** Analyze the command and provide a score that reflects the GREATEST LEVEL OF DETAIL. (e.g., Use 14 instead of 10, or 76 instead of 80).

    Command Text to Analyze: {clean_command_data}
    
    You MUST return ONLY a single JSON object. Do not include any other text, reasoning, or markdown formatting.
    JSON Schema: {{"S_Score": [integer from 0 to 100]}}
    """

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": LPAST_PROMPT}
            ],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        # Extract and Parse the Score
        raw_response = completion.choices[0].message.content
        cleaned_response = raw_response.strip()

        cleaned_response = re.sub(r'```json|```', '', cleaned_response, flags=re.IGNORECASE).strip()

        score_data = json.loads(cleaned_response)
        s_score = score_data.get("S_Score", 50)

        return int(s_score)

    except Exception as e:
        # Fallback to a neutral score on any API or parsing error
        st.warning(f"Groq API call failed for command: '{command_text}'. Final Error: {e}")
        return 50


def get_action_summary(prediction_tier: str, label_list: list) -> str:
    """
    Queries the LLM to generate a human-readable action plan based on the detected attack labels.
    """
    if groq_client is None:
        return "LLM Client Error: Cannot generate live intelligence summary."

    unique_labels = list(set(label_list))
    labels_str = ", ".join(unique_labels[:5])

    # Prepare the Prompt for the Threat Analyst Persona
    LPAST_ACTION_PROMPT = f"""
    You are a Senior Cyber Threat Analyst. Your task is to analyze a set of detected attack labels and provide a concise, actionable summary for the security operations team.

    PREDICTION TIER: {prediction_tier}

    DETECTED ATTACK LABELS (Top 5): {labels_str}

    TASK:
    1. Summarize the overall **Nature of the Threat** (e.g., "High-volume network congestion" or "Exploratory access").
    2. Define the **Immediate Action** required for this tier.

    You MUST return ONLY a concise text summary.
    """

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": LPAST_ACTION_PROMPT}],
            temperature=0.1,
            max_tokens=500,
        )

        return completion.choices[0].message.content

    except Exception as e:
        # Fallback error message
        return f"Error generating intelligence: API Failed. ({e})"

# =================================================================
# MAIN APPLICATION LOGIC
# =================================================================

st.set_page_config(layout="wide", page_title="L-PAST Project")
st.title("🛡️ L-PAST: LLM-Passive Attribution for Semantics & Timing")
st.markdown("Upload your prepared CSV containing the L-PAST features.")

uploaded_file = st.file_uploader(
    "Must contain: Variance, Label, and Ground_Truth",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # --- Data Validation ---
        required_cols = ['Variance', 'Label', 'Ground_Truth']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Error: The uploaded file must contain the prepared columns: {required_cols}")
            st.warning("Please ensure you have manually added the 'Ground_Truth' column.")
            st.stop()

        # --- Prediction Calculation ---
        with st.spinner('Calculating Attribution Scores...'):
            if 'S_Score' not in df.columns:
                df['S_Score'] = df['command_text'].apply(get_live_s_score)

            df['P_AI (%)'] = df.apply(
                lambda row: calculate_p_ai(row['S_Score'], row['Variance']), axis=1
            )

            # --- Determine Prediction ---
            df['Prediction'] = df['P_AI (%)'].apply(get_attribution)

        st.success("Analysis Complete!")

        # Normalize Ground Truth
        df['Ground_Truth_Normalized'] = df['Ground_Truth'].str.strip().str.upper()

        y_true = df['Ground_Truth'].apply(lambda x: 'AI-Automated' if x == 'AI-Automated' else 'Human-Led')

        y_pred_binary = df['Prediction'].apply(
            lambda x: 'AI-Automated' if x in ['AI-Automated', 'Machine-Assisted'] else 'Human-Led'
        )

        f1 = f1_score(y_true, y_pred_binary, pos_label='AI-Automated', zero_division=0)
        acc = accuracy_score(y_true, y_pred_binary)

        # --- Metrics ---
        st.header("L-PAST Evaluation")

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("Classification Scores")
            st.metric("F1-Score", f"{f1:.4f}")
            st.metric("Overall Prediction Accuracy", f"{acc:.4f}")

            # Confidence Metrics for display
            conf_counts = df['Prediction'].value_counts()
            ai_auto_count = conf_counts.get('AI-Automated', 0)
            machine_assisted_count = conf_counts.get('Machine-Assisted', 0)

            combined_threat_count = ai_auto_count + machine_assisted_count

            st.info(f"AI/Machine Threat (Combined): **{combined_threat_count} samples**")
            st.warning(f"Human-Led (Non-Threat): **{conf_counts.get('Human-Led', 0)} samples**")

            # Confidence Bar Chart
            fig_metrics, ax_metrics = plt.subplots(figsize=(6, 4))
            sns.barplot(
                x=conf_counts.index,
                y=conf_counts.values,
                palette=['#dc3545', '#ffc107', '#007bff'],
                ax=ax_metrics,
                order=['AI-Automated', 'Machine-Assisted', 'Human-Led']
            )
            ax_metrics.set_title("3-Tier Attribution Distribution")
            ax_metrics.set_ylabel("Count")
            ax_metrics.set_xlabel("")
            st.pyplot(fig_metrics)

        with col_right:
            st.subheader("Feature Space Validation")

            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))

            sns.scatterplot(
                x='S_Score',
                y='Variance',
                hue='Prediction',
                data=df,
                palette={'AI-Automated': 'red', 'Machine-Assisted': 'orange', 'Human-Led': 'blue'},
                s=70,
                edgecolor='w',
                ax=ax_scatter
            )

            S_boundary_95 = np.linspace(0, 60, 60)
            T_boundary_95 = 350000 * (1 - (S_boundary_95 / 60))
            S_boundary_50 = np.linspace(0, 20, 20)
            T_boundary_50 = 50000 * (1 - S_boundary_50 / 10)

            ax_scatter.plot(S_boundary_95, T_boundary_95, color='darkblue', linestyle='--', label='P_AI=95% Boundary')
            ax_scatter.plot(S_boundary_50, T_boundary_50, color='orange', linestyle='--', label='P_AI=50% Boundary')

            ax_scatter.xaxis.set_major_locator(ticker.MultipleLocator(10))
            ax_scatter.tick_params(axis='x', rotation=45)

            ax_scatter.set_title('L-PAST Feature Space: Semantic Entropy vs. Timing (3-Tier)')
            ax_scatter.set_xlabel('S-Score (Linguistic Noise)')
            ax_scatter.set_ylabel('Variance (Timing Delay)')
            ax_scatter.set_xlim(-5, 105)
            st.pyplot(fig_scatter)

        st.divider()


        # --- ACTIONABLE INTELLIGENCE SUMMARY ---
        st.header("Actionable Intelligence Summary")
        st.markdown("The predicted counts translate directly into mandatory defensive actions:")

        col_act1, col_act2, col_act3 = st.columns(3)
        conf_counts = df['Prediction'].value_counts()
        ai_auto_count = conf_counts.get('AI-Automated', 0)
        machine_assisted_count = conf_counts.get('Machine-Assisted', 0)
        human_led_count = conf_counts.get('Human-Led', 0)

        # Get list of labels for each prediction tier
        ai_auto_labels = df[df['Prediction'] == 'AI-Automated']['Label'].tolist()
        machine_assisted_labels = df[df['Prediction'] == 'Machine-Assisted']['Label'].tolist()
        human_led_labels = df[df['Prediction'] == 'Human-Led']['Label'].tolist()


        # Action 1: AI-Automated (Immediate Threat)
        with col_act1:
            st.subheader("Immediate Threat")
            st.markdown(f"**Count:** **{ai_auto_count} Detected**")
            st.write("---")

            with st.spinner("Generating Live Action Plan..."):
                ai_auto_summary = get_action_summary("Immediate Threat (AI-Automated)", ai_auto_labels)
                st.markdown(ai_auto_summary)

        # Action 2: Machine-Assisted (Evasive Threat)
        with col_act2:
            st.subheader("Evasive Threat")
            st.markdown(f"**Count:** **{machine_assisted_count} Detected**")
            st.write("---")

            with st.spinner("Generating Live Action Plan..."):
                ma_summary = get_action_summary("Evasive Threat (Machine-Assisted)", machine_assisted_labels)
                st.markdown(ma_summary)

        # Action 3: Human-Led (Non-Immediate Threat)
        with col_act3:
            st.subheader("Non-Immediate Threat")
            st.markdown(f"**Count:** **{human_led_count} Detected**")
            st.write("---")

            with st.spinner("Generating Live Action Plan..."):
                hl_summary = get_action_summary("Non-Immediate Threat (Human-Led)", human_led_labels)
                st.markdown(hl_summary)

        st.divider()

        # Attribution Table
        st.header("Detailed Attribution Table")

        def color_confidence_style(val):
            if val == 'AI-Automated':
                return 'background-color: #f8d7da'
            elif val == 'Machine-Assisted':
                return 'background-color: #f0cf6e'
            elif val == 'Human-Led':
                return 'background-color: #cce5ff'
            return None


        st.dataframe(
            df[['Label', 'Variance', 'S_Score', 'P_AI (%)', 'Prediction', 'Ground_Truth']]
            .style.applymap(color_confidence_style, subset=['Prediction']),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        st.warning(
            "Please check that your CSV is correctly delimited and contains the required columns: 'Variance', 'Label', and 'Ground_Truth'.")