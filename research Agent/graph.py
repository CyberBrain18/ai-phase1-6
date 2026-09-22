import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tavily import TavilyClient
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"])
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def web_search(query: str) -> str:
    """Searches the web and returns a summarised set of results"""
    results = tavily.search(query=query, search_depth="basic", max_results=3)
    
    formatted = []
    for r in results["results"]:
        formatted.append(f"Source: {r["title"]} ({r['url']})\n{r['content']}")
        
    return "\n\n".join(formatted)

class ResearchState(TypedDict):
    question: str
    sub_questions: List[str]
    findings: List[str]
    critique: str
    report: str
    report_critique: str
    revision_count: int
    
def planner_node(state: ResearchState) -> dict:
    prompt = f"""Break this research question into 2-3 focused subquestions that together would let someone answer the original question well.
    Question: {state['question']}
    Respond with a numbered list with one sub-question per line, no other text"""
    
    response = llm.invoke(prompt)
    raw = response.content.strip()
    
    sub_questions = [
        line.split(".", 1)[-1].strip()
        for line in raw.split("\n")
        if line.strip()
    ]
    
    return {"sub_questions": sub_questions}
    
def researcher_node(state: ResearchState) -> dict:
    print("[researcher_node starting]")
    all_findings = []
    
    for sub_q in state["sub_questions"]:
        search_results = web_search(sub_q)
    
        prompt = f"""Based on the searched results below, create a concise, factual summary anwering the specific question. Only use information given in the search results-do not add any outside knowledge. If the results dont clearly answer the question, say so.
        question: {sub_q}
        
        search_results: 
        {search_results}
        
        Summary:"""
        
        response = llm.invoke(prompt)
        all_findings.append(f"Q: {sub_q}\nA: {response.content.strip()}")
    return {"findings": all_findings}

# if __name__ == "__main__":
#     test_state: ResearchState = {
#         "question": "Should a seed-stage startup use AWS, GCP, or Azure for its infrastructure?",
#         "sub_questions": [],
#         "findings": [],
#         "critique": "",
#         "report": ""
#     }
#     planned = planner_node(test_state)
#     test_state.update(planned)

#     researched = researcher_node(test_state)
#     test_state.update(researched)

#     for f in test_state["findings"]:
#         print(f)
#         print("---")
        
def critic_node(state: ResearchState) -> dict:
    print("[critic_node starting]") 
    findings_text = "\n\n".join(state["findings"])
    
    prompt = f"""You are a critical reviewer of the research findings. Review the findings against the original question and sub_questions they were meant to answer.
    Check specifically for:
    1. off_topic findings - does each finding's answer actually address its own stated questions?(e.g. if the question asks about AWS/GCP/Azure but the answer discusses unrelated platforms,
    flag this explicitly.)
    2. unsupported claims - any specific numbers or facts that seem inserted without clear grounding
    in search results.
    3. GAPS — does anything in the original question remain unanswered by the findings overall?
   
    overall_question: {state['question']}
    Findings:
    {findings_text}

    If there are real issues, list them clearly and specifically (name which finding, and what's wrong).
    If there are no issues, respond with exactly: NO ISSUES FOUND"""
    
    response = llm.invoke(prompt)
    return {"critique": response.content.strip()}

# if __name__ == "__main__":
#     test_state: ResearchState = {
#         "question": "Should a seed-stage startup use AWS, GCP, or Azure for its infrastructure?",
#         "sub_questions": [],
#         "findings": [],
#         "critique": "",
#         "report": ""
#     }
#     planned = planner_node(test_state)
#     test_state.update(planned)

#     researched = researcher_node(test_state)
#     test_state.update(researched)

#     critiqued = critic_node(test_state)
#     test_state.update(critiqued)

#     print(test_state["critique"])

def writer_node(state: ResearchState) -> dict:
    print(f"[writer_node starting, revision_count so far: {state.get('revision_count', 0)}]")
    findings_text = "\n\n".join(state["findings"])
    
    prompt = f"""Write a clear, well-structured report answering the original question below, using the research findings provided.
    Take the reviewer's critique into account: where the critique flagged an issue (off-topic
    content, unsupported claims, or gaps), either address it directly, exclude the flagged content,
    or explicitly note the limitation in the report rather than silently repeating the same problem.

    Original question: {state['question']}

    Findings:
    {findings_text}

    Reviewer's critique:
    {state['critique']}
    Write the final report:"""
    response = llm.invoke(prompt)
    return {"report": response.content.strip()}

def report_critic_node(state: ResearchState) -> dict:
    print("[report_critic_node starting]")
    prompt = f"""You are reviewing a final research report for factual grounding.

    Check specifically for:
    1. FABRICATED SPECIFICS — any precise numbers, dollar figures, or percentages in the report
    that do NOT appear anywhere in the original findings below. Flag these explicitly.
    2. Anything else clearly ungrounded or invented.

    Original findings (the ONLY source of truth):
    {chr(10).join(state['findings'])}

    Final report to review:
    {state['report']}

    If there are real issues, list them specifically. If none, respond with exactly: NO ISSUES FOUND"""

    response = llm.invoke(prompt)
    return {"report_critique": response.content.strip(), "revision_count": state.get("revision_count", 0) + 1}

def should_revise(state: ResearchState) -> str:
    if state["revision_count"] >= 2:
        return "end"  # safety cap — never loop forever
    if "NO ISSUES FOUND" in state["report_critique"]:
        return "end"
    return "revise"

graph = StateGraph(ResearchState)

graph.add_node("planner", planner_node)
graph.add_node("researcher", researcher_node)
graph.add_node("critic", critic_node)
graph.add_node("writer", writer_node)
graph.add_node("report_critic", report_critic_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "researcher")
graph.add_edge("researcher", "critic")
graph.add_edge("critic", "writer")
graph.add_edge("writer", "report_critic")

graph.add_conditional_edges(
    "report_critic",
    should_revise,
    {"revise": "writer", "end": END}
)

app = graph.compile()

if __name__ == "__main__":
    initial_state: ResearchState = {
        "question": "Should a seed-stage startup use AWS, GCP, or Azure for its infrastructure?",
        "sub_questions": [],
        "findings": [],
        "critique": "",
        "report": "",
        "report_critique": "",
        "revision_count": 0
    }

    final_state = app.invoke(initial_state)

    print("=== FINAL REPORT ===")
    print(final_state["report"])
    print("\n=== REVISION COUNT ===")
    print(final_state["revision_count"])
    print("\n=== FINAL REPORT CRITIQUE ===")
    print(final_state["report_critique"])