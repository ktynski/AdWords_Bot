import streamlit as st
import anthropic
import pandas as pd
import concurrent.futures
import base64
import json
import requests
from bs4 import BeautifulSoup
import openai


# Set up Streamlit app
st.set_page_config(page_title="Keyword Analysis and Ad Copy Generation", layout="wide")
st.title("Keyword Analysis and Ad Copy Generation")

# Sidebar for user inputs
with st.sidebar:
    st.subheader("User Inputs")
    
    # Anthropic API key input
    anthropic_api_key = st.text_input("Enter your Anthropic API key:", type="password")
    openai_api_key = st.text_input("Enter your Anthropic API key:", type="password")
    openai = openai.Client(api_key=openai_api_key)

    # Keyword input
    input_type = st.radio("Select input type:", ("Manual", "CSV File"))
    if input_type == "Manual":

        default_keywords= """data destruction services
                            how to securely destroy digital data
                            data shredding companies near me
                            DIY methods for data destruction
                            data destruction certifications
                            best data destruction software
                            secure data destruction standards
                            hard drive destruction service cost
                            onsite vs offsite data destruction pros and cons
                            data destruction for GDPR compliance
                            how to destroy SSD data securely
                            data destruction equipment for sale
                            environmentally friendly data destruction
                            data destruction policies template
                            data erasure software reviews
                            cost of mobile data destruction services
                            how to verify data destruction
                            data destruction for healthcare industry
                            secure data destruction for financial services
                            physical vs digital data destruction techniques
                            data destruction audit checklist
                            data destruction business opportunities
                            data destruction regulations by country
                            secure data wipe open source tools
                            data destruction job description"""

        keywords_input = st.text_area("Enter keywords (one per line):", value=default_keywords)
        keywords = [keyword.strip() for keyword in keywords_input.split("\n") if keyword.strip()]
    else:
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            keywords_df = pd.read_csv(uploaded_file)
            keywords = keywords_df['keyword'].tolist()
        else:
            keywords = []
    
    # Landing page URL input
    landing_page_url = st.text_input("Enter the landing page URL:",value="https://synetictechnologies.com/data-destruction-services")
    
    # Persona input
    persona = st.text_area("Define the persona for the ad writer:",value="Purna Virji, World's most expert on Adwords Ads, ROI, and ad writing skills and tactics.")
    
    # Additional parameters
    st.subheader("Additional Parameters")
    max_retries = st.number_input("Max retries for ad copy generation", min_value=1, max_value=10, value=3)
    max_workers = st.number_input("Max concurrent workers", min_value=1, max_value=10, value=5)
    temperature = st.slider("Temperature for keyword analysis", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    ad_copy_temperature = st.slider("Temperature for ad copy generation", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
    num_ads = st.number_input("Number of ad copies to generate", min_value=1, max_value=10, value=5)
    
    # Model selection
    model_options = {
        "Claude (Opus)": "claude-3-opus-20240229",
        "Claude (Sonnet)": "claude-3-sonnet-20240229",
        "Claude (Haiku)": "claude-3-haiku-20240307"
    }
    selected_model = st.selectbox("Select the Claude model:", list(model_options.keys()))
    
    # Start button
    start_button = st.button("Start Keyword Analysis")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Landing Page Evaluation")
    if "landing_page_evaluation" not in st.session_state:
        st.session_state.landing_page_evaluation = ""
    st.write(st.session_state.landing_page_evaluation)

with col2:
    st.subheader("Progress")
    if "progress_text" not in st.session_state:
        st.session_state.progress_text = ""
    st.write(st.session_state.progress_text)

# Function to get the keyword analysis results in JSON format
def get_keyword_analysis_results(keyword,persona):
    prompt = f"""
    You Are: {persona}, You are an extremely skilled, experienced, and persuasive adwords marketer that specializes in high CTR adwords strategies that have very high ROIs and very high CTR because they capture attention in the right way at the right time for the right target audience.
    Objective: Analyze the given keyword and provide insights for Google Ads campaigns.
    Keyword: {keyword}
    Instructions:
    1. Determine the relevance score of the keyword to data destruction and hard drive shredding services on a scale of 1 to 10. Provide specific and detailed justifications highlighting the key aspects of the keyword that make it relevant to the target services.
    2. Identify the intent category and subcategory behind the keyword using the User Intent Taxonomy for Web Search. Consider the broader context and potential user goals when determining the intent. Provide justifications that capture the nuances of the keyword and user intent.
    3. Identify the stage in the buyer's journey that the keyword represents. Elaborate on the justifications and provide specific recommendations on how to tailor ad content and targeting based on the identified stage.
    4. Identify the psychographic profile that the keyword is most likely to appeal to. Focus on the most relevant psychographic traits and provide justifications that directly tie those traits to the keyword and target audience. Include specific examples or insights on how the psychographic profile can inform ad creation and targeting.
    5. Identify the key demographic factors to consider when targeting ads for the keyword. Consider a broader range of demographic factors and provide justifications that explain how each factor relates to the keyword and target audience. Prioritize the most impactful demographic factors for ad targeting.
    6. Estimate the expected cost-per-click (CPC) bucket for the keyword.
    7. Estimate the expected monthly search volume bucket for the keyword.
    8. Estimate the competition level for the keyword. Provide detailed justifications that compare the keyword's competition level to other related keywords or industries. Offer insights on how the competition level may impact ad strategies and bidding.
    9. Estimate the expected quality score for the keyword on a scale of 1 to 10.
    10. Recommend the most suitable match type for the keyword.
    11. Identify potential hidden opportunities associated with the keyword. Think creatively and provide unique, actionable opportunities specific to the keyword and target audience. Consider unconventional strategies or emerging trends that could give a competitive edge.
    12. Identify potential negative keyword patterns associated with the keyword.
    
    Provide the analysis results in JSON format with the following structure:
    {{
      "relevance_score": {{
        "score": <integer>,
        "justification": <string>
      }},
      "intent": {{
        "category": <string>,
        "subcategory": <string>,
        "justification": <string>
      }},
      "buyers_journey_stage": {{
        "stage": <string>,
        "justification": <string>
      }},
      "psychographic_profile": {{
        "profile": <string>,
        "justification": <string>
      }},
      "demographic_factors": [
        {{
          "factor": <string>,
          "justification": <string>
        }},
        ...
      ],
      "cpc_bucket": {{
        "bucket": <string>,
        "justification": <string>
      }},
      "volume_bucket": {{
        "bucket": <string>,
        "justification": <string>
      }},
      "competition_level": {{
        "level": <string>,
        "justification": <string>
      }},
      "quality_score": {{
        "score": <integer>,
        "justification": <string>
      }},
      "match_type_recommendation": {{
        "match_type": <string>,
        "justification": <string>
      }},
      "hidden_opportunities": {{
        "opportunities": <string>,
        "justification": <string>
      }},
      "negative_keyword_patterns": [<string>, ...]
    }}
    """
    try:
        result = anthropic.Anthropic(api_key=anthropic_api_key).messages.create(
            model=model_options[selected_model],
            system="You provide only the requested output and never any commentary, preamble or postamble, or user notes. You are an expert at analyzing keywords for Google Ads campaigns.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=temperature,
        )
        keyword_insights = json.loads(result.content[0].text.strip())
        return keyword_insights
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        # If the initial attempt fails, use GPT-4 to fix the JSON output
        gpt4_prompt = [
            {"role": "user", "content": f"Format the following text as a JSON object with the structure specified in the original prompt:\n\n{result.content[0].text.strip()}"}
        ]
        gpt4_response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=gpt4_prompt,
            max_tokens=2000,
            temperature=0.2,
        )
        json_output = gpt4_response.choices[0].message.content.strip()
        
        try:
            keyword_insights = json.loads(json_output)
            return keyword_insights
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            return {"error": f"Failed to analyze keyword. Error: {str(e)}"}
    
    return {"error": "Failed to analyze keyword due to an unknown error."}

# Function to get the ad copies for a keyword
def get_keyword_ad_copies(keyword, keyword_insights, landing_page_evaluation, persona, max_retries=3):
    prompt = f"""
    Persona: You are a world-renowned expert in crafting highly effective Google Ads copies. Your ad copies consistently outperform industry benchmarks and drive exceptional click-through rates and conversions.
    
    Ad Writer Persona: {persona}
    
    Best Practices:
    1. Conduct thorough keyword research to understand user intent and target audience.
    2. Create compelling headlines that grab attention and communicate the main benefit.
    3. Use strong calls-to-action that encourage users to take the desired action.
    4. Highlight unique selling points and key differentiators to stand out from competitors.
    5. Ensure ad copy aligns with the target landing page for a seamless user experience.
    6. Continuously test and optimize ad copies based on performance data.
    
    Objective: Generate {num_ads} expert-level Google Ads ad copies for the given keyword, incorporating insights from the keyword analysis and landing page evaluation.
    
    Keyword: {keyword}
    
    Keyword Insights:
    {json.dumps(keyword_insights, indent=2)}
    
    Landing Page Evaluation:
    {landing_page_evaluation}
    
    Instructions:
    1. Carefully review the keyword insights and landing page evaluation to understand the target audience, intent, competitive landscape, and landing page content.
    2. Break down the ad creation process into logical steps, considering headline creation, description crafting, and URL path selection.
    3. Generate {num_ads} ad copies that masterfully incorporate the keyword insights, align with the landing page content, and adhere to industry best practices for ad copywriting.
    4. Each ad copy should have the following components:
       - Headline 1 (max 30 characters)
       - Headline 2 (max 30 characters)
       - Headline 3 (max 30 characters)
       - Description 1 (max 90 characters)
       - Description 2 (max 90 characters)
       - Path 1 (max 15 characters)
       - Path 2 (max 15 characters)
    5. Ensure the ad copies are compelling, relevant, and optimized for click-through rates and conversions.
    6. Place strong emphasis on including powerful calls-to-action (CTAs) and highlighting unique value propositions (UVPs) in the ad copies. Refer to the examples provided for guidance on creating persuasive and differentiated ad copy variations.
    
    CTA Examples:
    - "Get Started Now"
    - "Limited Time Offer"
    - "Schedule Your Free Consultation"
    - "Secure Your Data Today"
    
    UVP Examples:
    - "Industry-Leading Data Destruction"
    - "Certified & Secure Hard Drive Shredding"
    - "Protect Your Sensitive Information"
    - "Compliant & Eco-Friendly Disposal"
    
    Provide the ad copies in JSON format with the following structure:
    [
      {{
        "headline1": <string>,
        "headline2": <string>,
        "headline3": <string>,
        "description1": <string>,
        "description2": <string>,
        "path1": <string>,
        "path2": <string>
      }},
      ...
    ]
    """
    
    try:
        result = anthropic.Anthropic(api_key=anthropic_api_key).messages.create(
            model=model_options[selected_model],
            system="You are an expert at writing compelling Google Ads ad copies that incorporate keyword insights and align with landing page content.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=ad_copy_temperature,
        )
        ad_copies = json.loads(result.content[0].text.strip())
        if len(ad_copies) == num_ads and all(
            all(key in ad_copy for key in ["headline1", "headline2", "headline3", "description1", "description2", "path1", "path2"])
            for ad_copy in ad_copies
        ):
            return ad_copies
        else:
            raise ValueError("Generated ad copies do not match the expected format.")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        # If the initial attempt fails, use GPT-4 to fix the JSON output
        gpt4_prompt = [
            {"role": "user", "content": f"Format the following text as a JSON array of {num_ads} ad copy objects, where each object has the keys 'headline1', 'headline2', 'headline3', 'description1', 'description2', 'path1', and 'path2':\n\n{result.content[0].text.strip()}"}
        ]
        gpt4_response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=gpt4_prompt,
            max_tokens=4000,
            temperature=0.2,
        )
        json_output = gpt4_response.choices[0].message.content.strip()
        
        try:
            ad_copies = json.loads(json_output)
            if len(ad_copies) == num_ads and all(
                all(key in ad_copy for key in ["headline1", "headline2", "headline3", "description1", "description2", "path1", "path2"])
                for ad_copy in ad_copies
            ):
                return ad_copies
            else:
                return [{"error": f"Failed to generate ad copies. Error: {str(e)}"}]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            return [{"error": f"Failed to generate ad copies. Error: {str(e)}"}]
    
    return [{"error": "Failed to generate ad copies due to an unknown error."}]

# Function to scrape and evaluate the landing page content
def evaluate_landing_page(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract relevant content from the landing page
        title = soup.title.string if soup.title else ""
        description = soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else ""
        headings = " ".join([heading.text for heading in soup.find_all(['h1', 'h2', 'h3'])])
        paragraphs = " ".join([p.text for p in soup.find_all('p')])
        
        # Prepare the prompt for evaluating the landing page content
        prompt = f"""
        Objective: Evaluate the given landing page content and provide insights for creating effective Google Ads ad copies.
        
        Landing Page URL: {url}
        
        Landing Page Content:
        Title: {title}
        Description: {description}
        Headings: {headings}
        Paragraphs: {paragraphs}
        
        Instructions:
        1. Analyze the landing page content to identify the main theme, unique selling points, and key benefits.
        2. Provide a summary of the landing page content and its relevance to the target audience.
        3. Suggest key elements or phrases from the landing page that can be incorporated into the ad copies to ensure alignment and consistency.
        4. Offer any additional insights or recommendations for creating ad copies based on the landing page content.
        
        Provide the evaluation results in the following format:
        Main Theme: <string>
        Unique Selling Points: <string>
        Key Benefits: <string>
        Summary: <string>
        Key Elements for Ad Copies: <string>
        Additional Insights: <string>
        """
        
        result = anthropic.Anthropic(api_key=anthropic_api_key).messages.create(
            model=model_options[selected_model],
            system="You are an expert at evaluating landing page content for creating effective Google Ads ad copies.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.5,
        )
        
        return result.content[0].text.strip()
    
    except Exception as e:
        return f"Error evaluating landing page: {str(e)}"

# Function to process a single keyword
def process_keyword(keyword, landing_page_evaluation, persona):
    keyword_insights = get_keyword_analysis_results(keyword, persona)
    st.session_state.progress_text = f"Analyzing keyword: {keyword}"
    ad_copies = get_keyword_ad_copies(keyword, keyword_insights, landing_page_evaluation, persona, max_retries)
    st.session_state.progress_text = f"Generating ad copies for keyword: {keyword}"
    return {
        "keyword": keyword,
        "relevance_score": keyword_insights.get("relevance_score", {}).get("score", "N/A"),
        "relevance_score_justification": keyword_insights.get("relevance_score", {}).get("justification", "N/A"),
        "intent_category": keyword_insights.get("intent", {}).get("category", "N/A"),
        "intent_subcategory": keyword_insights.get("intent", {}).get("subcategory", "N/A"),
        "intent_justification": keyword_insights.get("intent", {}).get("justification", "N/A"),
        "buyers_journey_stage": keyword_insights.get("buyers_journey_stage", {}).get("stage", "N/A"),
        "buyers_journey_stage_justification": keyword_insights.get("buyers_journey_stage", {}).get("justification", "N/A"),
        "psychographic_profile": keyword_insights.get("psychographic_profile", {}).get("profile", "N/A"),
        "psychographic_profile_justification": keyword_insights.get("psychographic_profile", {}).get("justification", "N/A"),
        "demographic_factors": ", ".join([factor.get("factor", "N/A") for factor in keyword_insights.get("demographic_factors", [])]),
        "demographic_factors_justification": ", ".join([factor.get("justification", "N/A") for factor in keyword_insights.get("demographic_factors", [])]),
        "cpc_bucket": keyword_insights.get("cpc_bucket", {}).get("bucket", "N/A"),
        "cpc_bucket_justification": keyword_insights.get("cpc_bucket", {}).get("justification", "N/A"),
        "volume_bucket": keyword_insights.get("volume_bucket", {}).get("bucket", "N/A"),
        "volume_bucket_justification": keyword_insights.get("volume_bucket", {}).get("justification", "N/A"),
        "competition_level": keyword_insights.get("competition_level", {}).get("level", "N/A"),
        "competition_level_justification": keyword_insights.get("competition_level", {}).get("justification", "N/A"),
        "quality_score": keyword_insights.get("quality_score", {}).get("score", "N/A"),
        "quality_score_justification": keyword_insights.get("quality_score", {}).get("justification", "N/A"),
        "match_type_recommendation": keyword_insights.get("match_type_recommendation", {}).get("match_type", "N/A"),
        "match_type_recommendation_justification": keyword_insights.get("match_type_recommendation", {}).get("justification", "N/A"),
        "hidden_opportunities": keyword_insights.get("hidden_opportunities", {}).get("opportunities", "N/A"),
        "hidden_opportunities_justification": keyword_insights.get("hidden_opportunities", {}).get("justification", "N/A"),
        "negative_keyword_patterns": ", ".join(keyword_insights.get("negative_keyword_patterns", [])),
        "ad_copies": json.dumps(ad_copies)
    }

# Function to process keywords
def process_keywords(keywords, landing_page_evaluation, persona):
    if "progress_text" not in st.session_state:
        st.session_state.progress_text = ""
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_keyword, keyword, landing_page_evaluation, persona) for keyword in keywords]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            results.append(future.result())
            st.session_state.progress_text = f"Processed {i+1}/{len(keywords)} keywords"
    
    return results

# Process keywords when the start button is clicked
if start_button and anthropic_api_key and keywords and landing_page_url:
    with st.spinner("Evaluating landing page content..."):
        landing_page_evaluation = evaluate_landing_page(landing_page_url)
    st.session_state.landing_page_evaluation = landing_page_evaluation
    
    with st.spinner("Processing keywords..."):
        results = process_keywords(keywords, landing_page_evaluation, persona)
    
    st.session_state.progress_text = f"Processed {len(keywords)}/{len(keywords)} keywords"
    st.success("Keyword processing completed!")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Display the DataFrame
    st.subheader("Keyword Analysis Results")
    st.dataframe(df)
    
    # Download button for the results (CSV)
    csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="keyword_analysis_results.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
else:
    st.warning("Please provide your Anthropic API key, enter keywords, provide the landing page URL, and click the 'Start Keyword Analysis' button to begin.")
