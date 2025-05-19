from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from graph import Graph # Assuming Graph is defined in graph.py
import logging
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI # Using ChatOpenAI as an example
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Disable logging for openai and httpx packages
logging.getLogger('openai').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Initialize LLM (replace with your preferred model)
# Ensure OPENAI_API_KEY or equivalent is set in your environment variables
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

class GraphState(TypedDict):
    """
    Represents the state of the documentation workflow graph.

    Attributes:
        outer_cfg: The outer control flow graph (JCL steps and COBOL programs).
        inner_cfgs: A dictionary of inner control flow graphs (COBOL paragraphs).
        nodes_to_process: A list of node IDs remaining to be documented (includes JCL steps, COBOL programs, and paragraphs).
        documented_nodes: A dictionary to store documentation for each node.
        current_node_id: The ID of the node currently being processed.
        output_dir: The directory to save documentation files.
        previous_node_documentation: Documentation of the previously processed node for context.
    """
    outer_cfg: Graph
    inner_cfgs: Dict[str, Graph]
    nodes_to_process: List[str]
    documented_nodes: Dict[str, Any]
    current_node_id: str | None
    output_dir: str
    previous_node_documentation: str | None

def initialize_state(outer_cfg: Graph, inner_cfgs: Dict[str, Graph], output_dir: str) -> GraphState:
    """
    Initializes the state for the LangGraph workflow.
    Nodes to process will include JCL steps, COBOL programs, and individual paragraphs.
    """
    nodes_for_processing_detailed = []

    # Add JCL steps and COBOL program IDs from outer_cfg
    for node_id in outer_cfg.nodes():
        node_data = outer_cfg._graph.nodes[node_id]
        nodes_for_processing_detailed.append(node_id)  # Add JCL steps and Program IDs

        # If it's a program and has an inner CFG, add its paragraphs
        if node_data.get('type') == 'program' and node_id in inner_cfgs:
            program_inner_cfg = inner_cfgs[node_id]
            # Sort paragraphs by name for consistency
            sorted_paragraph_nodes = sorted(list(program_inner_cfg.nodes()))
            for para_node_id in sorted_paragraph_nodes:  # para_node_id is like "PROG1:PARA_A"
                nodes_for_processing_detailed.append(para_node_id)

    # Assuming IDs are unique enough
    nodes_to_process_final = nodes_for_processing_detailed

    logger.info(f"Initializing state with {len(nodes_to_process_final)} detailed nodes to process.")
    return {
        "outer_cfg": outer_cfg,
        "inner_cfgs": inner_cfgs,
        "nodes_to_process": nodes_to_process_final,
        "documented_nodes": {},
        "current_node_id": None,
        "output_dir": output_dir,
        "previous_node_documentation": None,
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
    Invokes the agent to document the current node (JCL step, Program, or Paragraph).
    """
    current_node_id = state["current_node_id"]
    outer_cfg = state["outer_cfg"]
    inner_cfgs = state["inner_cfgs"]
    previous_doc = state.get("previous_node_documentation")

    if not current_node_id:
        logger.warning("document_current_node called with no current_node_id.")
        return {}

    logger.info(f"Documenting node: {current_node_id}")

    node_attributes = {}
    node_type = 'unknown'
    is_paragraph_node = ':' in current_node_id  # Simple check: "PROG_ID:PARA_NAME"

    if is_paragraph_node:
        program_id, paragraph_name = current_node_id.split(':', 1)
        if program_id in inner_cfgs and current_node_id in inner_cfgs[program_id].nodes():
            node_attributes = inner_cfgs[program_id]._graph.nodes[current_node_id]
            node_type = node_attributes.get('type', 'paragraph')  # Should be 'paragraph'
        else:
            logger.error(f"Paragraph node {current_node_id} not found in inner CFGs.")
            # Create some default documentation or skip
            return {"documented_nodes": state["documented_nodes"], "previous_node_documentation": f"Error: Paragraph node {current_node_id} details not found."}
    elif current_node_id in outer_cfg.nodes():
        node_attributes = outer_cfg._graph.nodes[current_node_id]
        node_type = node_attributes.get('type', 'unknown')
    else:
        logger.error(f"Node {current_node_id} not found in outer or inner CFGs.")
        return {"documented_nodes": state["documented_nodes"], "previous_node_documentation": f"Error: Node {current_node_id} details not found."}

    # Construct prompt based on node type and available attributes
    documentation_prompt = f"""
    You are a documentation agent. Your task is to generate a concise documentation summary for a component in a legacy system.
    The overall objective is to understand the entire application's structure and behavior by documenting each piece.
    Each component is a unit of executable code or a structural element.

    Component ID: {current_node_id}
    Component Type: {node_type}

    Based on the provided information, generate a markdown summary. Focus on:
    - What this component represents (JCL step, COBOL program, COBOL paragraph, etc.)
    - Key attributes or its specific code
    - Its purpose or function within the larger system or program
    - Connections to other components (e.g., JCL executes Program, Program calls Subprogram, Paragraph PERFORMs another Paragraph)

    Documentation for the PREVIOUSLY PROCESSED component:
    ---
    {previous_doc if previous_doc else 'This is the first component being documented or no previous documentation is available.'}
    ---

    Current Component Details:
    """

    if node_type == 'jcl_step':
        job_name = node_attributes.get('jobName', 'N/A') # Get the job name
        step_name = node_attributes.get('stepName', 'N/A') # Get the step name
        step_number = node_attributes.get('stepNumber', 'N/A') # Get the step number
        datasets = node_attributes.get('datasets', [])
        code_with_comments = node_attributes.get('codeWithComments', 'N/A')
        code_without_comments = node_attributes.get('codeWithoutComments', 'N/A')
        documentation_prompt += f"""
    Parent JCL Job: {job_name}
    Step Name: {step_name}
    Step Number: {step_number}
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
        code_with_comments = node_attributes.get('code_with_comments', 'N/A')
        code_without_comments = node_attributes.get('code_without_comments', 'N/A')
        has_inner_cfg = node_attributes.get('has_inner_cfg', False)

        documentation_prompt += f"""
    This is a COBOL Program.
    Key Divisions:
    Identification Division: {identification_division}
    Environment Division: {environment_division}
    Data Division: {data_division}
    Procedure Division Using: {procedure_division_using}
    Full Program Code (all paragraphs):
    ```cobol
    {code_with_comments}
    ```
    Code (without comments):
    ```cobol
    {code_without_comments}
    ```
    Inner Control Flow Graph Available: {has_inner_cfg}
    This program might contain several paragraphs. Detailed documentation for each paragraph will follow if applicable.
    """
    elif node_type == 'paragraph':
        # Access attributes from the inner_cfg node
        para_code_with_comments = node_attributes.get('code_with_comments', 'N/A')
        para_name = node_attributes.get('name', 'Unknown Paragraph')
        program_id = current_node_id.split(':', 1)[0] if ':' in current_node_id else 'Unknown Program'
        
        documentation_prompt += f"""
    This is a COBOL Paragraph named '{para_name}' within program '{program_id}'.
    Paragraph Code:
    ```cobol
    {para_code_with_comments}
    ```
    """

    documentation_prompt += """

    Keep it brief and easy to understand.
    """
    
    print("Documentation Prompt:\n", documentation_prompt)  # Commented out for production

    try:
        # Invoke the LLM to generate documentation
        response = await llm.ainvoke([HumanMessage(content=documentation_prompt)])
        documentation = response.content
        logger.info(f"Generated documentation for {current_node_id}: {documentation}")
    except Exception as e:
        logger.error(f"Error generating documentation for {current_node_id}: {e}")
        documentation = f"Error generating documentation: {e}"

    # Create a copy of the dictionary to properly update LangGraph state
    documented_nodes = state["documented_nodes"].copy()
    documented_nodes[current_node_id] = documentation

    return {"documented_nodes": documented_nodes, "previous_node_documentation": documentation}

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