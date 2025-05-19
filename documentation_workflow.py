from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from graph import Graph # Assuming Graph is defined in graph.py
import logging
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI # Using ChatOpenAI as an example
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

logger = logging.getLogger(__name__)

# Initialize LLM (replace with your preferred model)
# Ensure OPENAI_API_KEY or equivalent is set in your environment variables
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

class GraphState(TypedDict):
    """
    Represents the state of the documentation workflow graph.

    Attributes:
        outer_cfg: The outer control flow graph.
        inner_cfgs: A dictionary of inner control flow graphs.
        nodes_to_process: A list of node IDs remaining to be documented.
        documented_nodes: A dictionary to store documentation for each node.
        current_node_id: The ID of the node currently being processed.
        output_dir: The directory to save documentation files.
    """
    outer_cfg: Graph
    inner_cfgs: Dict[str, Graph]
    nodes_to_process: List[str]
    documented_nodes: Dict[str, Any]
    current_node_id: str | None
    output_dir: str

def initialize_state(outer_cfg: Graph, inner_cfgs: Dict[str, Graph], output_dir: str) -> GraphState:
    """
    Initializes the state for the LangGraph workflow.
    """
    all_nodes = list(outer_cfg.nodes())
    # Add inner graph nodes if needed, depending on documentation granularity
    # For now, let's focus on outer graph nodes (JCL steps and programs)
    logger.info(f"Initializing state with {len(all_nodes)} nodes to process.")
    return {
        "outer_cfg": outer_cfg,
        "inner_cfgs": inner_cfgs,
        "nodes_to_process": all_nodes,
        "documented_nodes": {},
        "current_node_id": None,
        "output_dir": output_dir,
    }

def select_next_node(state: GraphState) -> GraphState:
    """
    Selects the next node to process from the list.
    """
    nodes_to_process = state["nodes_to_process"]
    if not nodes_to_process:
        logger.info("No more nodes to process.")
        return {"current_node_id": None}

    next_node_id = nodes_to_process.pop(0)
    logger.info(f"Selected next node: {next_node_id}")
    return {
        "nodes_to_process": nodes_to_process,
        "current_node_id": next_node_id,
    }

async def document_current_node(state: GraphState) -> GraphState:
    """
    Invokes the agent to document the current node.
    """
    current_node_id = state["current_node_id"]
    outer_cfg = state["outer_cfg"]
    inner_cfgs = state["inner_cfgs"]

    if not current_node_id:
        logger.warning("document_current_node called with no current_node_id.")
        return {}

    logger.info(f"Documenting node: {current_node_id}")

    node_attributes = outer_cfg._graph.nodes[current_node_id]
    node_type = node_attributes.get('type', 'unknown')

    # Construct prompt based on node type and available attributes
    documentation_prompt = f"""
    You are a documentation agent. Your task is to generate a concise documentation summary for a node in a legacy system graph.

    Node ID: {current_node_id}
    Node Type: {node_type}

    Based on the provided information, generate a markdown summary. Focus on:
    - What this node represents (JCL step, COBOL program, etc.)
    - Key attributes
    - Any relevant code snippets (if available)
    - Connections to other nodes (e.g., JCL executes Program, Program calls Subprogram)

    Node Details:
    """

    if node_type == 'jcl_step':
        datasets = node_attributes.get('datasets', [])
        code_with_comments = node_attributes.get('codeWithComments', 'N/A')
        code_without_comments = node_attributes.get('codeWithoutComments', 'N/A')
        documentation_prompt += f"""
    Datasets: {datasets}
    Code (with comments):
    ```jcl
{code_with_comments}
    ```
    Code (without comments):
    ```jcl
{code_without_comments}
    ```
    """
    elif node_type == 'program':
        identification_division = node_attributes.get('identification_division', {})
        environment_division = node_attributes.get('environment_division', {})
        data_division = node_attributes.get('data_division', {})
        procedure_division_using = node_attributes.get('procedure_division_using', [])
        code_with_comments = node_attributes.get('codeWithComments', 'N/A')
        code_without_comments = node_attributes.get('codeWithoutComments', 'N/A')
        has_inner_cfg = node_attributes.get('has_inner_cfg', False)

        documentation_prompt += f"""
    Identification Division: {identification_division}
    Environment Division: {environment_division}
    Data Division: {data_division}
    Procedure Division Using: {procedure_division_using}
    Code (with comments):
    ```cobol
{code_with_comments}
    ```
    Code (without comments):
    ```cobol
{code_without_comments}
    ```
    Inner Control Flow Graph Available: {has_inner_cfg}
    """

    documentation_prompt += """

    Keep it brief and easy to understand.
    """

    try:
        # Invoke the LLM to generate documentation
        response = await llm.ainvoke([HumanMessage(content=documentation_prompt)])
        documentation = response.content
        logger.info(f"Generated documentation for {current_node_id}")
    except Exception as e:
        logger.error(f"Error generating documentation for {current_node_id}: {e}")
        documentation = f"Error generating documentation: {e}"

    documented_nodes = state["documented_nodes"]
    documented_nodes[current_node_id] = documentation

    return {"documented_nodes": documented_nodes}

def save_documentation(state: GraphState) -> GraphState:
    """
    Saves the generated documentation to files.
    """
    documented_nodes = state["documented_nodes"]
    output_dir = state["output_dir"]
    logger.info(f"Saving documentation for {len(documented_nodes)} nodes to {output_dir}.")

    os.makedirs(output_dir, exist_ok=True)

    for node_id, doc_content in documented_nodes.items():
        # Sanitize node_id for filename
        filename = f"{node_id.replace(':', '_').replace('/', '_')}.md"
        file_path = os.path.join(output_dir, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc_content)
            logger.debug(f"Saved documentation for {node_id} to {file_path}")
        except Exception as e:
            logger.error(f"Error saving documentation for {node_id} to {file_path}: {e}")

    logger.info("Documentation saving complete.")
    return {}

def should_continue(state: GraphState) -> str:
    """
    Determines if there are more nodes to process.
    """
    if state.get("current_node_id") is not None:
        # If a node was just selected, always continue to document it
        return "document_node"
    elif state["nodes_to_process"]:
        # If no node was selected but nodes_to_process is not empty, select next
        return "select_next_node"
    else:
        # No more nodes to process
        return END

# Build the LangGraph workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("select_next_node", select_next_node)
workflow.add_node("document_node", document_current_node)
workflow.add_node("save_documentation", save_documentation)

# Define edges
workflow.add_edge(START, "select_next_node")
workflow.add_conditional_edges(
    "select_next_node",
    should_continue,
    {
        "document_node": "document_node",
        END: "save_documentation", # Transition to save when no more nodes
    },
)
workflow.add_edge("document_node", "select_next_node") # After documenting, select next
workflow.add_edge("save_documentation", END)

# Compile the graph
# Compilation should ideally happen in main.py where the workflow is invoked
# app = workflow.compile()

# Example usage (will be moved to main.py)
# if __name__ == "__main__":
#     # Assume outer_cfg and inner_cfgs are built
#     # from main.py import outer_cfg, inner_cfgs
#     # initial_state = initialize_state(outer_cfg, inner_cfgs)
#     # result = app.invoke(initial_state)
#     # print("Workflow finished.")
#     pass