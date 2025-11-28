import json
import os
import re
from typing import TypedDict, Dict, Any, List, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from operator import add
from dotenv import load_dotenv

# Import our custom tools
from tools import query_pubmed_realtime, search_jan_aushadhi

# --- CONFIGURATION ---
load_dotenv()

# Initialize Gemini
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("❌ GOOGLE_API_KEY missing in .env")

# --- CRITICAL FIX: SAFETY SETTINGS ---
# Medical queries often trigger "Dangerous Content" filters. We must relax them for a clinical tool.
safety_settings = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}

# Main reasoning model (Deterministic)
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", 
    temperature=0,
    safety_settings=safety_settings
)

# Summarization model
summarizer_llm = ChatGoogleGenerativeAI(
    model="gemini-pro", 
    temperature=0.2,
    safety_settings=safety_settings
)

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    # Thread logic
    patient_id: str
    doctor_query: str
    
    # Context logic
    patient_data: Dict[str, Any]
    research_evidence: str
    jan_aushadhi_result: Dict[str, Any]
    
    # Memory logic
    summary: str 
    final_response: Dict[str, Any]

# ==========================================
# 🕵 NODE 1: PERSONALIZATION AGENT
# ==========================================
def personalization_agent(state: AgentState):
    """
    Fetches patient profile.
    Optimization: If patient data is already loaded in memory, skip fetching.
    """
    if state.get("patient_data") and "name" in state["patient_data"]:
        print("👤 Personalization Agent: Data already in memory.")
        return {} # No state update needed

    print(f"👤 Personalization Agent: Loading profile for {state['patient_id']}...")
    try:
        with open("data/patients.json", "r") as f:
            all_patients = json.load(f)
        
        patient = all_patients.get(state['patient_id'])
        if not patient:
            return {"patient_data": {"error": "Patient not found"}}
            
        return {"patient_data": patient}
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return {"patient_data": {}}

# ==========================================
# 🔬 NODE 2: RESEARCH AGENT (Context Aware)
# ==========================================
def research_agent(state: AgentState):
    """
    Fetches evidence. Uses 'summary' to understand context of follow-up questions.
    """
    query = state['doctor_query']
    patient_context = state['patient_data'].get('condition_tags', [])
    current_summary = state.get("summary", "")
    
    print(f"🔬 Research Agent: Analyzing '{query}'...")
    
    # 1. Search Jan Aushadhi (Cost Check)
    jan_result = search_jan_aushadhi(query)
    
    # 2. Search PubMed (Safety Check)
    # Enrich query with patient context for better results
    refined_query = f"{query} {current_summary} {' '.join(patient_context)} contraindications guidelines"
    
    evidence = query_pubmed_realtime(refined_query)
    
    return {
        "research_evidence": evidence,
        "jan_aushadhi_result": jan_result
    }

# ==========================================
# 🛡 NODE 3: SAFETY AGENT (The Decider)
# ==========================================
def safety_agent(state: AgentState):
    """
    Synthesizes data to make a decision.
    """
    print("🛡 Safety Agent: Validating treatment plan...")
    
    prompt = f"""
    ACT AS: Med Perplexity, an expert Clinical Safety Architect.
    
    --- SESSION CONTEXT (MEMORY) ---
    PREVIOUS DISCUSSION: {state.get('summary', 'No prior context.')}
    
    --- CURRENT INPUTS ---
    PATIENT: {json.dumps(state['patient_data'], indent=2)}
    DOCTOR QUERY: "{state['doctor_query']}"
    PUBMED EVIDENCE: {state['research_evidence']}
    JAN AUSHADHI: {json.dumps(state['jan_aushadhi_result'], indent=2)}
    
    --- TASK ---
    Analyze the order for SAFETY and COST EFFICIENCY considering the patient's history.
    If the doctor asks a follow-up question, answer it using the Evidence provided.
    
    --- REQUIRED JSON OUTPUT ---
    {{
      "status": "approved" | "blocked" | "caution" | "info",
      "title": "Short Headline",
      "message": "Clear explanation.",
      "evidence": "Source citation.",
      "suggestion": "Alternative if blocked.",
      "savings": {{ "found": true/false, "text": "..." }}
    }}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # --- ROBUST JSON EXTRACTION ---
        # Gemini might return "Here is the JSON: json ... "
        # We use Regex to extract ONLY the JSON part { ... }
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        
        if json_match:
            clean_text = json_match.group(0)
            final_decision = json.loads(clean_text)
        else:
            raise ValueError("No JSON block found in LLM response")
        
        # Jan Aushadhi Math Fallback (Reliability Guard)
        if state['jan_aushadhi_result'].get('found') and final_decision['status'] != 'blocked':
             tool_data = state['jan_aushadhi_result']['drug_data']
             final_decision['savings'] = {
                 "found": True,
                 "text": f"Save {tool_data['savings_amount']} by switching to Jan Aushadhi {tool_data['generic_name']}."
             }
        
        return {"final_response": final_decision}
        
    except Exception as e:
        print(f"❌ Safety Agent Error: {e}")
        print(f"DEBUG - LLM RAW RESPONSE: {response.content if 'response' in locals() else 'None'}")
        
        return {"final_response": {
            "status": "caution", 
            "message": "Error processing safety check. Please review guidelines manually.", 
            "title": "System Error"
        }}

# ==========================================
# 📝 NODE 4: SUMMARIZER (The Memory Keeper)
# ==========================================
def summarizer_agent(state: AgentState):
    """
    Condenses the current interaction into a summary string for the next turn.
    """
    print("📝 Summarizer: Updating clinical context...")
    
    current_summary = state.get("summary", "")
    last_query = state['doctor_query']
    last_decision = state['final_response']
    
    prompt = f"""
    Update the Clinical Session Summary based on the latest interaction.
    Keep it concise but medically precise.
    
    OLD SUMMARY: {current_summary}
    
    LATEST INTERACTION:
    Doctor: {last_query}
    AI Decision: {last_decision.get('status')} - {last_decision.get('message')}
    
    NEW SUMMARY:
    """
    
    response = summarizer_llm.invoke([HumanMessage(content=prompt)])
    return {"summary": response.content}

# ==========================================
# 🕸 BUILD THE GRAPH
# ==========================================
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("personalize", personalization_agent)
workflow.add_node("research", research_agent)
workflow.add_node("safety", safety_agent)
workflow.add_node("summarize", summarizer_agent)

# Define Edges
workflow.set_entry_point("personalize")
workflow.add_edge("personalize", "research")
workflow.add_edge("research", "safety")
workflow.add_edge("safety", "summarize") # Update memory after decision
workflow.add_edge("summarize", END)

# Add Memory Persistence
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ==========================================
# 🧪 TEST ZONE (Run this file directly)
# ==========================================
if __name__ == "__main__":
    import pprint

    # Config for the thread (Session ID)
    config = {"configurable": {"thread_id": "session_1"}}

    print("\n🏥 --- STARTING MED PERPLEXITY MEMORY TEST ---")

    # TURN 1: The Trap
    print("\n🗣 TURN 1: 'Prescribe Diclofenac for Rahul'")
    input_1 = {
        "patient_id": "PAT-001",
        "doctor_query": "Prescribe Diclofenac for joint pain",
        "summary": "" # Start empty
    }
    
    # Invoke with config to save state
    for event in app.stream(input_1, config=config):
        pass # Just run it
    
    # Get final state
    final_state_1 = app.get_state(config)
    print("🤖 AI RESPONSE 1:")
    pprint.pprint(final_state_1.values['final_response'])
    print(f"📝 MEMORY UPDATED: {final_state_1.values['summary']}")

    # TURN 2: The Follow-up (Context Aware)
    print("\n🗣 TURN 2: 'What about Paracetamol?' (Implicitly referring to Rahul)")
    input_2 = {
        "patient_id": "PAT-001", # Same patient
        "doctor_query": "What about Paracetamol?",
        # Notice we don't pass 'summary' here; LangGraph fetches it from memory!
    }
    
    for event in app.stream(input_2, config=config):
        pass
        
    final_state_2 = app.get_state(config)
    print("🤖 AI RESPONSE 2:")
    pprint.pprint(final_state_2.values['final_response'])
    print(f"📝 MEMORY UPDATED: {final_state_2.values['summary']}")