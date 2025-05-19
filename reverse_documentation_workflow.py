from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from graph import Graph  # Assuming Graph is defined in graph.py
import logging
import os
import networkx as nx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI  # Using ChatOpenAI as an example
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Disable logging for openai and httpx packages
logging.getLogger('openai').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Initialize LLM (replace with your preferred model)
# Ensure OPENAI_API_KEY or equivalent is set in your environment variables
llm = ChatOpenAI(model="o4-mini", reasoning_effort="high")

class ReverseDocumentationState(TypedDict):
    """
    Represents the state for the reverse documentation workflow.
    
    Attributes:
        outer_cfg: The outer control flow graph (JCL steps and COBOL programs).
        inner_cfgs: A dictionary of inner control flow graphs (COBOL paragraphs).
        outer_nodes_to_process: List of node IDs to document in reverse order.
        current_outer_node_id: The ID of the node currently being processed.
        generated_documentation: Dictionary mapping node IDs to their documentation.
        final_docs_to_save: Dictionary mapping node IDs to their final documentation.
        output_dir: The directory to save documentation files.
    """
    outer_cfg: Graph
    inner_cfgs: Dict[str, Graph]
    outer_nodes_to_process: List[str]  # In reverse order
    current_outer_node_id: Optional[str]
    generated_documentation: Dict[str, str]  # {node_id: documentation}
    final_docs_to_save: Dict[str, str]  # {node_id: final_documentation}
    output_dir: str

def get_reverse_processing_order(outer_cfg: Graph) -> List[str]:
    """
    Determine a reverse processing order for outer CFG nodes.
    Ideally a reverse topological sort, but fallback to simple reversal if needed.
    """
    logger.info("Determining reverse processing order for outer CFG nodes...")
    
    try:
        # Try to get a reverse topological sort if graph is a DAG
        # This might fail if there are cycles in the graph
        topo_sorted = list(nx.topological_sort(outer_cfg._graph))
        reversed_order = list(reversed(topo_sorted))
        logger.info(f"Using reverse topological order with {len(reversed_order)} nodes")
        return reversed_order
    except nx.NetworkXUnfeasible:
        # If not a DAG, fall back to a simple reverse of the node list
        logger.warning("Graph is not a DAG. Using simple reversed node list instead.")
        
        # Process programs first, then JCL steps in reverse order
        program_ids = [nid for nid, data in outer_cfg.nodes(data=True) 
                     if data.get('type') == 'program']
        jcl_step_ids = [nid for nid, data in outer_cfg.nodes(data=True) 
                      if data.get('type') == 'jcl_step']
        
        reversed_order = list(reversed(program_ids)) + list(reversed(jcl_step_ids))
        logger.info(f"Using custom reverse order with {len(reversed_order)} nodes")
        return reversed_order

async def document_cobol_program_in_reverse(program_id: str, inner_cfg: Graph, llm: ChatOpenAI, 
                                         outer_context: str) -> str:
    """
    Document a COBOL program by processing its paragraphs in reverse execution order.
    
    Args:
        program_id: The ID of the COBOL program
        inner_cfg: The inner CFG containing paragraphs
        llm: The LLM to use for generating documentation
        outer_context: Context from subsequent system components
        
    Returns:
        Aggregated documentation for the program
    """
    # Get paragraphs in reverse order (either topological if possible, or simple reverse)
    paragraph_node_ids = [nid for nid in inner_cfg.nodes() 
                        if nid.startswith(f"{program_id}:")]  
    
    try:
        # Try reverse topological sort for paragraphs
        para_sorted = list(nx.topological_sort(inner_cfg._graph.subgraph(paragraph_node_ids)))
        paragraph_node_ids = list(reversed(para_sorted))
    except nx.NetworkXUnfeasible:
        # Simple reverse if not a DAG
        logger.warning(f"Inner CFG for {program_id} is not a DAG. Using simple reversed node list.")
        paragraph_node_ids.reverse()
    
    logger.info(f"Documenting {len(paragraph_node_ids)} paragraphs in {program_id} in reverse order")
    
    aggregated_paragraph_docs = []
    current_inner_context = ""  # Documentation of paragraphs already processed in reverse
    
    for para_node_id in paragraph_node_ids:
        node_attributes = inner_cfg._graph.nodes[para_node_id]
        para_name = node_attributes.get('name', para_node_id.split(":")[-1])
        para_code = node_attributes.get('code_with_comments', 'No code found for paragraph.')
        
        prompt = f"""
        You are documenting COBOL program '{program_id}' by analyzing its paragraphs in REVERSE order of execution.
        
        The overall system context (components that execute AFTER this program) is:
        ---
        {outer_context if outer_context else 'No subsequent system context provided.'}
        ---

        You are currently at paragraph: '{para_name}'.
        Its code is:
        ```cobol
        {para_code}
        ```

        The documentation for paragraphs that execute AFTER '{para_name}' (which you have already processed in this reverse pass) is:
        ---
        {current_inner_context if current_inner_context else f"This is the last paragraph being analyzed in reverse order, so no documentation exists yet for subsequent paragraphs."}
        ---

        Based on the code, the subsequent program logic (inner context), and the subsequent system logic (outer context),
        describe the purpose and key actions of paragraph '{para_name}'.
        Focus on how it contributes to achieving the effects documented in the subsequent paragraphs/system components.
        Be concise (3-5 sentences) for this single paragraph.
        """
        
        # Don't log the entire prompt to avoid excessive log size
        logger.info(f"Generating documentation for paragraph '{para_name}' in program '{program_id}'")
        
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            para_doc = response.content
            
            # Insert at beginning to maintain original execution order in final doc
            aggregated_paragraph_docs.insert(0, f"### Paragraph: {para_name}\n{para_doc}")
            
            # Update context for next paragraph (previous in execution flow)
            current_inner_context = "\n\n".join(aggregated_paragraph_docs)
            
            logger.info(f"Successfully documented paragraph '{para_name}' in program '{program_id}'")
        except Exception as e:
            logger.error(f"Error documenting paragraph {para_name}: {e}")
            aggregated_paragraph_docs.insert(0, f"### Paragraph: {para_name}\nError generating documentation: {e}")
    
    # Create final program documentation with all paragraphs in correct execution order
    final_program_doc = f"# Documentation for COBOL Program: {program_id}\n\n"
    final_program_doc += "*This documentation was generated by analyzing paragraphs in reverse execution order, from program end to program start.*\n\n"
    final_program_doc += "\n\n".join(aggregated_paragraph_docs)
    
    return final_program_doc

def initialize_reverse_workflow(outer_cfg: Graph, inner_cfgs: Dict[str, Graph], 
                              output_dir: str) -> ReverseDocumentationState:
    """
    Initialize the state for the reverse documentation workflow.
    """
    reversed_outer_nodes = get_reverse_processing_order(outer_cfg)
    
    logger.info(f"Initializing reverse documentation workflow with {len(reversed_outer_nodes)} nodes")
    return {
        "outer_cfg": outer_cfg,
        "inner_cfgs": inner_cfgs,
        "outer_nodes_to_process": reversed_outer_nodes,
        "current_outer_node_id": None,
        "generated_documentation": {},
        "final_docs_to_save": {},
        "output_dir": output_dir
    }

def select_next_outer_node(state: ReverseDocumentationState) -> Dict[str, Any]:
    """
    Select the next node from the reverse-ordered list to process.
    """
    nodes_to_process = state["outer_nodes_to_process"]
    if not nodes_to_process:
        logger.info("No more nodes to process in reverse order.")
        return {"current_outer_node_id": None}

    next_node_id = nodes_to_process.pop(0)  # Get from front of the reversed list
    logger.info(f"Selected next node for reverse processing: {next_node_id}")
    return {
        "outer_nodes_to_process": nodes_to_process,
        "current_outer_node_id": next_node_id,
    }

async def generate_documentation_for_outer_node(state: ReverseDocumentationState) -> Dict[str, Any]:
    """
    Generate documentation for the current outer node, using context from already processed "future" nodes.
    """
    current_outer_node_id = state["current_outer_node_id"]
    if not current_outer_node_id:
        return {}

    outer_cfg = state["outer_cfg"]
    inner_cfgs = state["inner_cfgs"]
    
    node_attributes = outer_cfg._graph.nodes[current_outer_node_id]
    node_type = node_attributes.get('type', 'unknown')

    # Build context from "future" nodes (those already processed in this reverse pass)
    context_from_future_nodes = "\n\n--- Context from Subsequent System Components ---\n"
    if not state["generated_documentation"]:
        context_from_future_nodes += "This appears to be the last component in the system flow (first being documented in reverse).\n"
    else:
        # Find direct successors if possible, otherwise use all processed nodes
        successors = list(outer_cfg._graph.successors(current_outer_node_id)) if current_outer_node_id in outer_cfg.nodes() else []
        if successors:
            # Use only direct successors that have been processed
            for succ_id in successors:
                if succ_id in state["generated_documentation"]:
                    context_from_future_nodes += f"\n## Component: {succ_id}\n{state['generated_documentation'][succ_id]}\n---\n"
        else:
            # If no direct successors or they haven't been processed, use the last few processed nodes
            processed_nodes = list(state["generated_documentation"].items())[-3:]  # Limit to avoid context explosion
            for node_id, doc in processed_nodes:
                context_from_future_nodes += f"\n## Component: {node_id}\n{doc}\n---\n"

    final_doc = ""

    if node_type == 'program':
        program_id = current_outer_node_id
        # Check if this is a placeholder node (program referenced but not available in parsed data)
        is_placeholder = node_attributes.get('is_placeholder', False)
        
        if is_placeholder:
            # Generate documentation for placeholder program
            note = node_attributes.get('note', 'No additional information available')
            prompt = f"""
            You are documenting a program '{program_id}' that is part of a legacy mainframe system.
            
            This program is referenced by other components but its source code is not available in the parsed data.
            Note: {note}
            
            The system components that execute AFTER this program are described as:
            ---
            {context_from_future_nodes}
            ---
            
            Generate a brief documentation summary for this program, focusing on:
            1. What can be inferred about its purpose based on its name and the components that call it
            2. How it relates to the components that execute after it
            3. Note that this is a placeholder documentation as the source code is not available
            
            Use a clear structure with markdown formatting.
            """
            
            logger.info(f"Generating placeholder documentation for program '{program_id}' which has no source code")
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            final_doc = response.content
            logger.info(f"Successfully generated placeholder documentation for program '{program_id}'")
            
        elif program_id in inner_cfgs:
            inner_cfg_for_program = inner_cfgs[program_id]
            logger.info(f"Generating documentation for COBOL program '{program_id}' by reverse inner traversal")
            program_doc = await document_cobol_program_in_reverse(
                program_id, inner_cfg_for_program, llm, context_from_future_nodes
            )
            final_doc = program_doc
        else:
            logger.warning(f"No inner CFG found for program {program_id}. Generating basic doc.")
            # Basic documentation if no inner CFG is available
            identification_division = node_attributes.get('identification_division', {})
            environment_division = node_attributes.get('environment_division', {})
            data_division = node_attributes.get('data_division', {})
            code_with_comments = node_attributes.get('code_with_comments', 'N/A')
            
            prompt = f"""
            You are documenting a COBOL program '{program_id}' that is part of a legacy mainframe system.
            
            The system components that execute AFTER this program are described as:
            ---
            {context_from_future_nodes}
            ---
            
            Program Details:
            Identification Division: {identification_division}
            Environment Division: {environment_division}
            Data Division: {data_division}
            Code:
            ```cobol
            {code_with_comments}
            ```
            
            Generate a comprehensive documentation summary for this program, focusing on:
            1. Its purpose and main function in the system
            2. How it relates to the components that execute after it
            3. Key data processing logic
            
            Use a clear structure with markdown formatting.
            """
            
            # Don't log the entire prompt to avoid excessive log size
            logger.info(f"Generating documentation for program '{program_id}' using context from subsequent nodes")
            
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            final_doc = response.content
            logger.info(f"Successfully generated documentation for program '{program_id}'")

    elif node_type == 'jcl_step':
        job_name = node_attributes.get('jobName', 'N/A')
        step_name = node_attributes.get('stepName', 'N/A')
        step_number = node_attributes.get('stepNumber', 'N/A')
        code_with_comments = node_attributes.get('codeWithComments', 'N/A')
        
        prompt = f"""
        You are documenting a JCL step '{current_outer_node_id}' that is part of a legacy mainframe system.
        
        The system components that execute AFTER this JCL step are described as:
        ---
        {context_from_future_nodes}
        ---
        
        JCL Step Details:
        Parent JCL Job: {job_name}
        Step Name: {step_name}
        Step Number: {step_number}
        Code:
        ```jcl
        {code_with_comments}
        ```
        
        Generate a comprehensive documentation summary for this JCL step, focusing on:
        1. Its purpose and main function in the job stream
        2. How it relates to the components that execute after it
        3. Key operations or data processed by this step
        
        Use a clear structure with markdown formatting.
        """
        
        # Don't log the entire prompt to avoid excessive log size
        logger.info(f"Generating documentation for JCL step '{current_outer_node_id}' using context from subsequent nodes")
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        final_doc = response.content
        logger.info(f"Successfully generated documentation for JCL step '{current_outer_node_id}'")
    else:
        logger.warning(f"Unsupported node type '{node_type}' for node {current_outer_node_id}")
        final_doc = f"# Documentation for {current_outer_node_id} (Type: {node_type})\n\nUnsupported component type.\n"

    # Update the documentation dictionaries
    generated_documentation = state["generated_documentation"].copy()
    generated_documentation[current_outer_node_id] = final_doc
    
    final_docs_to_save = state["final_docs_to_save"].copy()
    final_docs_to_save[current_outer_node_id] = final_doc

    return {
        "generated_documentation": generated_documentation, 
        "final_docs_to_save": final_docs_to_save
    }

def save_all_documentation(state: ReverseDocumentationState) -> Dict[str, Any]:
    """
    Save all the generated documentation to files.
    """
    output_dir = state["output_dir"]
    final_docs = state["final_docs_to_save"]
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving documentation for {len(final_docs)} components to {output_dir}")
    
    for doc_id, doc_content in final_docs.items():
        # Sanitize ID for filename
        filename = f"{doc_id.replace(':', '_').replace('/', '_')}.md"
        file_path = os.path.join(output_dir, filename)
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc_content)
            logger.info(f"Saved documentation for {doc_id} to {file_path}")
        except Exception as e:
            logger.error(f"Error saving documentation for {doc_id}: {e}")
    
    return {}

def should_continue_processing(state: ReverseDocumentationState) -> str:
    """
    Determine if we should continue processing nodes.
    """
    if state.get("current_outer_node_id") is None and not state["outer_nodes_to_process"]:
        return END  # No current node and no more nodes to process
    
    # If we have a current node to document or more nodes to process
    if state.get("current_outer_node_id") is not None:
        return "document_node"  # Document the current node
    elif state["outer_nodes_to_process"]:
        return "select_next_node"  # Select another node
    
    return END  # Fallback

# Build the LangGraph workflow
reverse_workflow = StateGraph(ReverseDocumentationState)

# Add nodes
reverse_workflow.add_node("select_next_node", select_next_outer_node)
reverse_workflow.add_node("document_node", generate_documentation_for_outer_node)
reverse_workflow.add_node("save_documentation", save_all_documentation)

# Define edges
reverse_workflow.add_edge(START, "select_next_node")

# Conditional edges based on current state
reverse_workflow.add_conditional_edges(
    "select_next_node",
    lambda state: "document_node" if state.get("current_outer_node_id") else END,
    {
        "document_node": "document_node",
        END: "save_documentation",
    },
)

reverse_workflow.add_edge("document_node", "select_next_node")  # Loop back for next node
reverse_workflow.add_edge("save_documentation", END)

# Compile the workflow
compiled_workflow = reverse_workflow.compile()

# Example of how to run the workflow
async def run_reverse_documentation(outer_cfg: Graph, inner_cfgs: Dict[str, Graph], output_dir: str):
    """
    Run the reverse documentation workflow.
    """
    initial_state = initialize_reverse_workflow(outer_cfg, inner_cfgs, output_dir)
    result = await compiled_workflow.ainvoke(initial_state)
    logger.info("Reverse documentation workflow completed successfully")
    return result