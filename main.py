import json
import re
import logging
import gradio as gr
from typing import Optional

from dotenv import dotenv_values, set_key
from openai import AzureOpenAI
from pydantic import BaseModel, Field, ValidationError


logging.basicConfig(
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
)

CONTEXT = """I'm going to give you a document that contains comments about a slate by the evaluating committee.
A slate is the list of nominees and the supporting documentation for proposed new members 
of a chartered study section. Nominees are not selected individually, but rather as a set to form a panel that will provide the 
fair and expert review.

The Primary requirements for slates:
 1. Scientific expertise
    • Members should collectively cover the topics and methods listed in the study section 
guidelines.
    • Individuals with broad expertise are valuable
 2. Multidimensional Diversity
    • Scientific diversity (no single “pedigree” or perspective on a problem)
    • Institutional
    • Personal demographics: gender, race, ethnicity
    • Career stage, review experience \n"""

IDEAL_PARAMETERS = """The ideal percentage of data for each criterion is given in json format below.
{
	"URM" : "20%",
	"Minority": "50%",
	"Female": "10%",

	"EA": "30%",
	"SO": "30%",
	"CE": "15%",
	"WE": "25%",
	"FO": "5%",

	"professor": "70%",
	"associate professor": "20%",
	"assistant professor": "10%"
}\n"""

TASK = """The deviation from the ideal values should be considered to see whether the criterion is satisfied or not.	

The demographic diversity criteria include URM, Minority and Female,
The geographic diversity criteria include EA, SO, CE, WE, and FO.
The seniority/career phase criterion include professor,associate professor,and assistant professor.

For each of the criterion given in the json format,do the following two operations:
1. Check whether any plan of action is stated in the workflow data given in <document> and <\document> if the actual and ideal values are deviated.
2. Find the sentiment for the plan of action as "Positive" or "Negative". 
If actual and ideal data are same, give: Plan of action as "Criterion satisfied" and sentiment as "Healthy". In case there is a plan of action given in workflow, provide that information. Also, 
If there is no plan of action provided for a deviant criterion, give plan of action as "No information provided", sentiment:"Negative".
3.The rating value for healthy, positive and negative are 3, 2 and 1 respectively.
4. Calculate the average rating of the demographic diversity, geographic diversity, seniority/career phase criterion and overall average.
5. Based on the Plan of action of each criterion, Summarise your analysis.

The format of your overall response should look like what's shown between the <example> tags. Make sure to follow the formatting and spacing exactly.
Answer immediately without preamble, do not include anything other than the json in your final response.

<example>
{
    "Demographic Diversity": {
        "URM": {
            "plan of action": "",
            "sentiment": "",
            "rating": ""
          },
          "Minority": {
            "plan of action": "",
            "sentiment": "",
            "rating": ""

          },
          "Female": {
            "plan of action": "",
            "sentiment": "",
            "rating": ""
          },
          "Average Rating": ""
    },
    "Geographic Diversity": {
      "EA": {
        "plan of action": "",
        "sentiment": "",
        "rating": ""
      },
      "SO": {
        "plan of action": "",
        "sentiment": "",
        "rating": ""
      },
      "CE": {
        "plan of action": "",
        "sentiment": "",
        "rating": ""
      },
      "WE": {
        "plan of action": "",
        "sentiment": "",
        "rating": ""
      },
      "FO": {
        "plan of action": "",
        "sentiment": "",
        "rating": ""
      },
      "Average Rating": ""
    },
    "Seniority/Career Phase": {
      "professor": {
        "plan of action": "",
        "sentiment": "",
        "rating": ""
      },
      "associate professor": {
        "plan of action": "",
        "sentiment": "",
        "rating": ""
      },
      "assistant professor": {
        "plan of action": "",
        "sentiment": "",
        "rating": ""
      },
      "Average Rating": ""
    },
    "Overall Rating": "",
    "Summary" : ""
  }
</example>


The workflow is given below in <document> and <\document> \n"""


def structure_response(response_dict: dict) -> list:
    """Structure the response from OpenAI to display in gradio

    Args:
        response (dict): Response from OpenAI

    Returns:
        list: List structured in order to display in gradio
    """

    out_list = []

    # Overall Rating

    out_overall = response_dict["Overall Rating"]
    out_list.append(out_overall)

    out_summary = response_dict["Summary"]
    out_list.append(out_summary)

    # Demographic Ratings

    out_demo_rating = response_dict["Demographic Diversity"]["Average Rating"]
    out_list.append(out_demo_rating)

    out_urm_rating = response_dict["Demographic Diversity"]["URM"]["rating"]
    out_list.append(out_urm_rating)

    out_urm_reason = response_dict["Demographic Diversity"]["URM"]["plan of action"]
    out_list.append(out_urm_reason)

    out_min_rating = response_dict["Demographic Diversity"]["Minority"]["rating"]
    out_list.append(out_min_rating)

    out_min_reason = response_dict["Demographic Diversity"]["Minority"][
        "plan of action"
    ]
    out_list.append(out_min_reason)

    out_fem_rating = response_dict["Demographic Diversity"]["Female"]["rating"]
    out_list.append(out_fem_rating)

    out_fem_reason = response_dict["Demographic Diversity"]["Female"]["plan of action"]
    out_list.append(out_fem_reason)

    # Geographic Ratings

    out_geo_rating = response_dict["Geographic Diversity"]["Average Rating"]
    out_list.append(out_geo_rating)

    out_ea_rating = response_dict["Geographic Diversity"]["EA"]["rating"]
    out_list.append(out_ea_rating)

    out_ea_reason = response_dict["Geographic Diversity"]["EA"]["plan of action"]
    out_list.append(out_ea_reason)

    out_so_rating = response_dict["Geographic Diversity"]["SO"]["rating"]
    out_list.append(out_so_rating)

    out_so_reason = response_dict["Geographic Diversity"]["SO"]["plan of action"]
    out_list.append(out_so_reason)

    out_ce_rating = response_dict["Geographic Diversity"]["CE"]["rating"]
    out_list.append(out_ce_rating)

    out_ce_reason = response_dict["Geographic Diversity"]["CE"]["plan of action"]
    out_list.append(out_ce_reason)

    out_we_rating = response_dict["Geographic Diversity"]["WE"]["rating"]
    out_list.append(out_we_rating)

    out_we_reason = response_dict["Geographic Diversity"]["WE"]["plan of action"]
    out_list.append(out_we_reason)

    out_fo_rating = response_dict["Geographic Diversity"]["FO"]["rating"]
    out_list.append(out_fo_rating)

    out_fo_reason = response_dict["Geographic Diversity"]["FO"]["plan of action"]
    out_list.append(out_fo_reason)

    # Seniority/Career Phase Ratings

    out_sen_rating = response_dict["Seniority/Career Phase"]["Average Rating"]
    out_list.append(out_sen_rating)

    out_prof_rating = response_dict["Seniority/Career Phase"]["professor"]["rating"]
    out_list.append(out_prof_rating)

    out_prof_reason = response_dict["Seniority/Career Phase"]["professor"][
        "plan of action"
    ]
    out_list.append(out_prof_reason)

    out_ass_prof_rating = response_dict["Seniority/Career Phase"][
        "associate professor"
    ]["rating"]
    out_list.append(out_ass_prof_rating)

    out_ass_prof_reason = response_dict["Seniority/Career Phase"][
        "associate professor"
    ]["plan of action"]
    out_list.append(out_ass_prof_reason)

    out_assis_prof_rating = response_dict["Seniority/Career Phase"][
        "assistant professor"
    ]["rating"]
    out_list.append(out_assis_prof_rating)

    out_assis_prof_reason = response_dict["Seniority/Career Phase"][
        "assistant professor"
    ]["plan of action"]
    out_list.append(out_assis_prof_reason)

    logging.info(f"List for gradio {out_list}")

    return out_list


def validate_response(response_str: str) -> bool:
    """function to validate json response from OpenAI, Returns True if AI response was structured correctly.

    Args:
        response_str (str): Response from OpenAI

    Returns:
        bool: True/False
    """

    class ActionRating(BaseModel):
        plan_of_action: Optional[str] = Field("plan of action", alias="plan of action")
        sentiment: Optional[str] = Field("sentiment", alias="sentiment")
        rating: float | None = Field("rating", alias="rating")

    class DemographicDiversity(BaseModel):
        URM: ActionRating
        Minority: ActionRating
        Female: ActionRating
        Average_Rating: float | None = Field("Average Rating", alias="Average Rating")

    class GeographicDiversity(BaseModel):
        EA: ActionRating
        SO: ActionRating
        CE: ActionRating
        WE: ActionRating
        FO: ActionRating
        Average_Rating: float | None = Field("Average Rating", alias="Average Rating")

    class SeniorityCareerPhase(BaseModel):
        professor: ActionRating = Field("Professor", alias="professor")
        associate_professor: ActionRating = Field(
            "Associate Professor", alias="associate professor"
        )
        assistant_professor: ActionRating = Field(
            "Assistant Professor", alias="assistant professor"
        )
        Average_Rating: float | None = Field("Average Rating", alias="Average Rating")

    class ResponseModel(BaseModel):
        Demographic_Diversity: DemographicDiversity = Field(
            "Demographic Diversity", alias="Demographic Diversity"
        )
        Geographic_Diversity: GeographicDiversity = Field(
            "Geographic Diversity", alias="Geographic Diversity"
        )
        Seniority_Career_Phase: SeniorityCareerPhase = Field(
            "Seniority/Career Phase", alias="Seniority/Career Phase"
        )
        Overall_Rating: float | None = Field("Overall Rating", alias="Overall Rating")
        Summary: Optional[str] = Field("Summary", alias="Summary")

    try:
        ResponseModel.model_validate_json(response_str)
        return True
    except ValidationError as e:
        logging.error("Failed to validate response from OpenAI", exc_info=True)
        logging.error(e)
        return False


def get_analysis(messages: list) -> str:
    """Calls the OpenAI Completions API, takes a list of message objects as input and returns AI response.

    Args:
        messages (list): List of message objects

    Returns:
        str: response from OpenAI model
    """

    secrets = dotenv_values(".env")
    API_KEY = secrets["AZURE_OPENAI_KEY"]
    DEPLOYMENT = secrets["DEPLOYMENT"]
    ENDPOINT = secrets["AZURE_OPENAI_ENDPOINT"]

    client = AzureOpenAI(
        api_key=API_KEY, api_version="2023-12-01-preview", azure_endpoint=ENDPOINT
    )
    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        temperature=0,
    )
    response_content = res.choices[0].message.content
    match = re.search(r"```json\n(.*?)```", response_content, re.DOTALL)

    if match:
        response_content = match.group(1).strip()

    logging.info(f"Response from OpenAI {response_content}")

    return response_content


def predict_slate_health(
    urm: str,
    minority: str,
    female: str,
    ea: str,
    so: str,
    ce: str,
    we: str,
    fo: str,
    prof: str,
    aprof: str,
    assisprof: str,
    workflow: str,
) -> list:
    """Analyses the Slate Workflow comments and compares the diversity ratios using OpenAI

    Args:
        urm (str): URM %
        minority (str): Minority %
        female (str): Female %
        ea (str): EA %
        so (str): SO %
        ce (str): CE %
        we (str): WE %
        fo (str): FO %
        prof (str): Professor %
        aprof (str): Associate Professor %
        assisprof (str): Assistant Professor %
        workflow (str): Workflow Comment

    Returns:
        list: list of rating and analysis for each criteria of a given slate
    """
    diversity_demographics = f"""The percentage of data for each criterion are given in json format below.
{{
	"URM" : "{urm}&",
	"Minority": "{minority}%",
	"Female": "{female}%",

	"EA": "{ea}%",
	"SO": "{so}%",
	"CE": "{ce}%",
	"WE": "{we}%",
	"FO": "{fo}%",

	"professor": "{prof}%",
	"associate professor": "{aprof}%",
	"assistant professor": "{assisprof}%"
}}\n"""

    document = f"""<document>{workflow}</document>"""

    prompt = f"""
{CONTEXT}
{IDEAL_PARAMETERS}
{diversity_demographics}
{TASK}
{document}
"""
    messages = [
        {
            "role": "system",
            "content": "You are an analyst for the Center for Scientific Review committee.",
        },
        {"role": "user", "content": prompt},
    ]

    response = get_analysis(messages)

    retries = 0

    while (not validate_response(response)) and retries < 3:
        logging.info("AI response was not properly structured")
        retries += 1
        messages.append({"role": "assistant", "content": response})
        messages.append(
            {
                "role": "user",
                "content": "Looks like the output you gave is not properly formatted. Can you modify your response to match the example I gave",
            }
        )
        response = get_analysis(messages)

    response_dict = json.loads(response)

    return structure_response(response_dict)


def add_key(key: str, deployment: str, endpoint: str) -> object:
    """Updates Environment variables from Gradio UI

    Args:
        key (str): OpenAI API Key
        deployment (str): Azure OpenAI Deployment Name
        endpoint (str): Azure OpenAI Endpoint url

    Returns:
        _type_: Gradio Object??
    """
    env_file = ".env"
    env_vars = dotenv_values(env_file)
    env_vars["AZURE_OPENAI_KEY"] = key
    env_vars["DEPLOYMENT"] = deployment
    env_vars["AZURE_OPENAI_ENDPOINT"] = endpoint

    try:
        for k, v in env_vars.items():
            set_key(env_file, k, v)
        result = "Key added successfully."
        return gr.Info(result)
    except Exception as e:
        logging.error("Failed to update environment variables", exc_info=True)
        return gr.Info(str(e))
