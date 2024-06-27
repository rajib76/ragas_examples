# This is an example to see if LLM can judge its own output
#
import os
from typing import Tuple

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


def _score_summary(summary: str, source: str):

    COMPREHENSIVENESS_SYSTEM_PROMPT_TMPLT = """
    You are tasked with evaluating summarization quality. Please follow the instructions below.

    INSTRUCTIONS:

    1. Identify the key points in the provided source text and assign them high or low importance level.

    2. Assess how well the summary captures these key points.

    Are the key points from the source text comprehensively included in the summary? More important key points matter more in the evaluation.

    Scoring criteria:
    0 - Capturing no key points with high importance level
    5 - Capturing 70 percent of key points with high importance level
    10 - Capturing all key points of high importance level

    Answer using the entire template below.

    TEMPLATE:
    Score: <The score from 0 (capturing none of the important key points) to 10 (captures all key points of high importance).>
    Criteria: <Mention key points from the source text that should be included in the summary>
    Supporting Evidence: <Which key points are present and which key points are absent in the summary.>"""

    COMPREHENSIVENESS_HUMAN_PROMPT_TMPLT = """
    /SOURCE TEXT/
    {source}
    /END OF SOURCE TEXT/

    /SUMMARY/
    {summary}
    /END OF SUMMARY/
    """

    COMPREHENSIVENESS_PROMPT = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    COMPREHENSIVENESS_SYSTEM_PROMPT_TMPLT
                )
            ),
            HumanMessagePromptTemplate.from_template(COMPREHENSIVENESS_HUMAN_PROMPT_TMPLT),
        ]
    )
    scorer_llm = ChatOpenAI(model_name="gpt-4")
    chain = COMPREHENSIVENESS_PROMPT | scorer_llm | StrOutputParser()

    score = chain.invoke({"source": source, "summary": summary})

    return score


def create_and_score_summary(source: str) -> Tuple[str, str]:
    summary_llm = ChatOpenAI(model_name="gpt-4")
    summary_llm.model_kwargs = {'temperature': 0.3, "max_tokens": 50}

    GENERATION_PROMPT_TMPLT = """Create a summary based on the provided source"""
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    GENERATION_PROMPT_TMPLT
                )
            ),
            HumanMessagePromptTemplate.from_template("{source}"),
        ]
    )

    chain = prompt | summary_llm | StrOutputParser()
    summary = chain.invoke({"source": source})
    score = _score_summary(summary,source)

    return summary, score


if __name__ == "__main__":
    source = """
    The Taj Mahal (/ˌtɑːdʒ məˈhɑːl, ˌtɑːʒ-/; lit. 'Crown of the Palace') is an ivory-white marble mausoleum on the right bank of the river Yamuna in Agra, Uttar Pradesh, India. It was commissioned in 1631 by the fifth Mughal emperor, Shah Jahan (r. 1628–1658) to house the tomb of his beloved wife, Mumtaz Mahal; it also houses the tomb of Shah Jahan himself. The tomb is the centrepiece of a 17-hectare (42-acre) complex, which includes a mosque and a guest house, and is set in formal gardens bounded on three sides by a crenellated wall.

    Construction of the mausoleum was completed in 1648, but work continued on other phases of the project for another five years. The first ceremony held at the mausoleum was an observance by Shah Jahan, on 6 February 1643, of the 12th anniversary of the death of Mumtaz Mahal. The Taj Mahal complex is believed to have been completed in its entirety in 1653 at a cost estimated at the time to be around ₹5 million, which in 2023 would be approximately ₹35 billion (US$77.8 million).

    The building complex incorporates the design traditions of Indo-Islamic and Mughal architecture. It employs symmetrical constructions with the usage of various shapes and symbols. While the mausoleum is constructed of white marble inlaid with semi-precious stones, red sandstone was used for other buildings in the complex similar to the Mughal era buildings of the time. The construction project employed more than 20,000 workers and artisans under the guidance of a board of architects led by Ustad Ahmad Lahori, the emperor's court architect.

    The Taj Mahal was designated as a UNESCO World Heritage Site in 1983 for being "the jewel of Islamic art in India and one of the universally admired masterpieces of the world's heritage". It is regarded as one of the best examples of Mughal architecture and a symbol of Indian history. The Taj Mahal is a major tourist attraction and attracts more than five million visitors a year. In 2007, it was declared a winner of the New 7 Wonders of the World initiative.
    """

    summary, score = create_and_score_summary(source)

    print("--- Summary of the source is: ---")
    print(summary)
    print("--- Score of the summary is: ---")
    print(score)
