from typing import Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

# ==========================================
# 1. DEFINE THE GRAPH STATE
# ==========================================
class AgentState(TypedDict):
    """The central state object tracked across the entire workflow."""
    anomaly_payload: Dict[str, Any]    # Raw telemetry metrics received from Kafka
    triage_analysis: str               # The LLM's classification of the problem
    runbook_instructions: str          # The context retrieved from the RAG Vector DB
    execution_command: str             # The exact shell command/script to execute
    execution_status: str              # 'SUCCESS', 'FAILED', or 'PENDING'
    retry_count: int                   # Track remote connection attempts

# ==========================================
# 2. DEFINE THE GRAPH NODES (Python Functions)
# ==========================================

def triage_anomaly_node(state: AgentState) -> Dict[str, Any]:
    """Node A: Analyzes the metrics and identifies the root issue."""
    print("\n🔍 [Node 1: Triage] Analyzing incoming telemetry payload...")
    metrics = state["anomaly_payload"].get("metrics", {})
    
    # In a full setup, you would pass these metrics into an LLM via LangChain here
    # For now, we simulate the logic parser:
    if metrics.get("memory_available_pct", 100) < 10.0:
        analysis = "CRITICAL_MEMORY_SHORTAGE"
    elif metrics.get("hardware_temperature_c", 0) > 85.0:
        analysis = "THERMAL_THROTTLING_WARNING"
    else:
        analysis = "UNKNOWN_PERFORMANCE_DEGRADATION"
        
    print(f"   Result: Identified problem profile as '{analysis}'")
    return {"triage_analysis": analysis, "retry_count": 0}


def rag_search_node(state: AgentState) -> Dict[str, Any]:
    """Node B: Queries the Vector Database for the matching runbook."""
    print("\n📚 [Node 2: RAG Search] Querying Vector DB for troubleshooting runbook...")
    problem_type = state["triage_analysis"]
    
    # Simulating a Vector DB retrieval lookup:
    if problem_type == "CRITICAL_MEMORY_SHORTAGE":
        runbook = "RUNBOOK_ID_402: High Memory Leak. Action: Clear cache buffers and kill non-system runaway tasks."
        command = "sudo sync && echo 3 > /proc/sys/vm/drop_caches"
    elif problem_type == "THERMAL_THROTTLING_WARNING":
        runbook = "RUNBOOK_ID_109: High Thermal Temp. Action: Throttle maximum background process CPU execution cap."
        command = "cpufreq-set -g powersave"
    else:
        runbook = "RUNBOOK_ID_000: Generic reset. Action: Log info and wait."
        command = "echo 'Monitoring anomaly logs...'"

    print(f"   Result: Retrieved {runbook}")
    return {"runbook_instructions": runbook, "execution_command": command}


def remote_execution_node(state: AgentState) -> Dict[str, Any]:
    """Node C: Connects to the Edge device and applies the automated fix."""
    current_retry = state.get("retry_count", 0) + 1
    print(f"\n⚙️ [Node 3: Execution] Attempting remote fix (Attempt #{current_retry})...")
    print(f"   Executing: `{state['execution_command']}` via remote secure channel...")
    
    # Simulating an edge connection issue on the first try to demonstrate LangGraph cycles
    if current_retry < 2:
        print("   ❌ Connection timed out! Edge device packet drops are too high.")
        return {"execution_status": "FAILED", "retry_count": current_retry}
    else:
        print("   ✅ Fix applied successfully! System telemetry stabilizing.")
        return {"execution_status": "SUCCESS", "retry_count": current_retry}

# ==========================================
# 3. DEFINE THE ROUTING EDGES (Conditional Logic)
# ==========================================

def evaluate_execution_status(state: AgentState) -> str:
    """Determines whether to retry the fix, log a failure, or exit successfully."""
    if state["execution_status"] == "SUCCESS":
        return "complete_workflow"
    
    if state["execution_status"] == "FAILED":
        if state["retry_count"] >= 3:
            print("\n🚨 [Alert] Maximum remote fix attempts reached. Escalating to human on-call.")
            return "escalate_and_abort"
        else:
            print("   🔄 Routing state back to Execution Node for a re-try loop...")
            return "retry_fix"
        
    return "complete_workflow"

# ==========================================
# 4. ASSEMBLE THE STATE GRAPH
# ==========================================

# Initialize the workflow graph configuration with our State definition
workflow = StateGraph(AgentState)

# Register the Python functions as usable workflow Nodes
workflow.add_node("triage_anomaly", triage_anomaly_node)
workflow.add_node("rag_search", rag_search_node)
workflow.add_node("remote_execution", remote_execution_node)

# Set the entry point of the state machine execution loop
workflow.set_entry_point("triage_anomaly")

# Create standard, deterministic paths (Edges)
workflow.add_edge("triage_anomaly", "rag_search")
workflow.add_edge("rag_search", "remote_execution")

# Create a conditional routing edge out of the execution phase
workflow.add_conditional_edges(
    "remote_execution",
    evaluate_execution_status,
    {
        "complete_workflow": END,
        "retry_fix": "remote_execution",
        "escalate_and_abort": END
    }
)

# Compile the graph architecture into an executable runtime engine
app = workflow.compile()

# ==========================================
# 5. TEST RUN THE WORKFLOW
# ==========================================
if __name__ == "__main__":
    # Mocking a high-value anomaly message coming from your TFLite container model
    mock_incoming_kafka_message = {
        "timestamp": "2026-07-14T22:00:00Z",
        "metrics": {
            "cpu_utilization_pct": 45.2,
            "memory_available_pct": 4.8, # Under 10% -> Will trigger Memory Shortage
            "disk_io_wait_ms": 12.0,
            "network_packet_drops": 4,
            "hardware_temperature_c": 52.0
        }
    }
    
    print("🎬 Starting LangGraph Orchestration Loop Engine...")
    
    # Run the graph synchronously using the mock event data
    final_output = app.invoke({"anomaly_payload": mock_incoming_kafka_message})
    
    print("\n🏁 [Graph Terminal State Reached]")
    print(f"Final Execution Summary Status: {final_output['execution_status']}")
