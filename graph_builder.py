# ================================
# Part 1: Graph Builder (NetworkX)
# ================================
import json
import logging
import networkx as nx
from dotenv import load_dotenv
from graph import Graph
load_dotenv()

logger = logging.getLogger(__name__)

def remove_edges(graph, edges):
    """
    Recursively remove edges that create cycles in the graph.
    
    Args:
        graph (Graph): The graph from which to remove edges.
        edges (list): A list of edges to remove.
        
    Returns:
        list: A list of removed edges.
    """
    removed_edges = []
    
    def find_cycle():
        """Find a cycle in the graph and return the last edge that creates it."""
        # Use NetworkX to find cycles in the graph
        try:
            cycle = nx.find_cycle(graph._graph, orientation='original')
            if cycle:
                # Get the last edge from the cycle
                u, v, _ = cycle[-1]
                return u, v
        except nx.NetworkXNoCycle:
            return None
    
    # Recursively identify and remove cycle-creating edges
    cycle_edge = find_cycle()
    while cycle_edge:
        u, v = cycle_edge
        # Remove the edge creating the cycle
        logger.debug("Removing edge %s -> %s to break cycle", u, v)
        graph.remove_edge(u, v)
        removed_edges.append((u, v))
        
        # Look for the next cycle
        cycle_edge = find_cycle()
    
    if removed_edges:
        logger.debug("Removed %d edges to break cycles", len(removed_edges))
    
    return removed_edges

def build_graphs(jcl_json_list, cobol_json_list):
    """
    Given lists of JCL and COBOL parser JSON objects, build:
      1. outer_cfg: a DiGraph of JCL -> COBOL and COBOL -> COBOL calls
      2. inner_cfg: a dict mapping programId to its inner CFG (paragraph-level)
    """    
    logger.info("Starting to build graphs from %d COBOL programs and %d JCL jobs", 
                len(cobol_json_list), len(jcl_json_list))
    outer_cfg = Graph()
    inner_cfg = {}
    
    # --- Process COBOL programs ---
    # Group paragraphs by program_id
    cobol_programs = {}
    for paragraph_json in cobol_json_list:
        prog_id = paragraph_json['program_id']
        if prog_id not in cobol_programs:
            cobol_programs[prog_id] = {
                'program_id': prog_id,
                'identification_division': paragraph_json.get('identification_division', {}),
                'environment_division': paragraph_json.get('environment_division', {}),
                'data_division': paragraph_json.get('data_division', {}),
                'procedure_division_using': paragraph_json.get('procedure_division', []).get('using', []),
                'paragraphs': {}
            }
        # Add paragraph details to the program
        paragraph_name = paragraph_json['procedure_division']['paragraph']['paragraph_name']
        cobol_programs[prog_id]['paragraphs'][paragraph_name] = paragraph_json

    for prog_id, prog_details in cobol_programs.items():
        logger.info("Processing COBOL program: %s", prog_id)
        paragraphs = prog_details['paragraphs']

        # ensure node exists in outer graph
        outer_cfg.add_node(prog_id, type='program',
                           has_inner_cfg=bool(paragraphs))
        
        all_called_programs = set()
        for p_name, p_details in paragraphs.items():
            # Collect called programs from all paragraphs
            for called_prog in p_details.get('called_programs', []):
                all_called_programs.add(called_prog)
        
        # build inner CFG for this program
        inner = Graph()
        logger.debug("Found %d paragraphs in program %s", len(paragraphs), prog_id)
        
        # add nodes
        for p,v in paragraphs.items():
            pid = f"{prog_id}:{p}"
            inner.add_node(pid,
                           name=p,
                           code_with_comments=v['procedure_division']['paragraph'].get('code_with_comments'),
                           code_without_comments=v.get('code_without_comments'),
                           type='paragraph')
        # add edges for PERFORM and GOTO targets
        for p,v in paragraphs.items():
            logger.debug("Processing paragraph %s", p)
            perform_targets = v['procedure_division']['paragraph'].get('perform_targets', [])
            goto_targets = v['procedure_division']['paragraph'].get('goto_targets', [])
            src = f"{prog_id}:{p}"
            for tgt in perform_targets:
                if tgt['target_name'] == 'INLINE': continue
                tgt_id = f"{prog_id}:{tgt['target_name']}"
                logger.debug("Adding PERFORM edge from %s to %s", src, tgt_id)
                inner.add_edge(src, tgt_id, type='PERFORM')
            for tgt in goto_targets:
                tgt_id = f"{prog_id}:{tgt}"
                logger.debug("Adding GOTO edge from %s to %s", src, tgt_id)
                inner.add_edge(src, tgt_id, type='GOTO')

        # remove dead code (no incoming except entry)
        entry = f"{prog_id}:ENTRY"
        dead_nodes = []
        # Check if ENTRY node exists before processing
        if entry in inner.nodes():
            for node in list(inner.nodes()):
                if node != entry and inner.in_degree(node) == 0:
                    dead_nodes.append(node)
                    inner.remove_node(node)
            if dead_nodes:
                logger.debug("Removed %d dead nodes from program %s: %s", 
                            len(dead_nodes), prog_id, dead_nodes)
        else:
             logger.warning(f"ENTRY paragraph not found for program {prog_id}. Skipping dead code removal.")

        # break cycles: recursively remove edges that create cycles
        removed_edges = remove_edges(inner, list(inner.edges()))
        # Edges have already been removed by the remove_edges function
        inner_cfg[prog_id] = inner
        
        # inter-program CALL edges
        called_programs_list = list(all_called_programs)
        if called_programs_list:
            logger.debug("Program %s calls %d other programs: %s", 
                        prog_id, len(called_programs_list), called_programs_list)
        for callee in called_programs_list:
            outer_cfg.add_node(callee, type='program')
            outer_cfg.add_edge(prog_id, callee, type='CALL')

    # --- Process JCL jobs ---
    for job in jcl_json_list:
        if 'jobName' not in job or 'step' not in job:
            logger.warning(f"Skipping JCL job due to missing 'jobName' or 'step' key: {job}")
            continue
        job_name = job['jobName']
        step = job['step']
        step_id = f"{job_name}:{step['stepName']}"
        logger.info("Processing JCL job step: %s", step_id)

        # Add node for the JCL job if it doesn't exist
        if job_name not in outer_cfg.nodes():
            outer_cfg.add_node(job_name, type='jcl_job', jobName=job_name)
            logger.debug(f"Added JCL job node: {job_name}")

        # Add node for the JCL step and link it to the job
        outer_cfg.add_node(step_id, type='jcl_step',
                           jobName=job_name,
                           stepName=step.get('stepName'),
                           stepNumber=step.get('stepNumber'),
                           datasets=step.get('datasets', []),
                           codeWithComments=step.get('codeWithComments', ''),
                           codeWithoutComments=step.get('codeWithoutComments', ''))

        # link JCL step->COBOL program
        prog_id = step['programId']
        # Ensure program node exists (might already exist from COBOL processing)
        if prog_id not in outer_cfg.nodes():
             outer_cfg.add_node(prog_id, type='program', isPlaceholder=True)
        outer_cfg.add_edge(step_id, prog_id, type='EXECUTES')

    logger.info("Graph building completed. Outer graph has %d nodes and %d edges.", 
               outer_cfg.number_of_nodes(), outer_cfg.number_of_edges())
    logger.info("Inner graph has %d nodes and %d edges.", 
               sum(inner.number_of_nodes() for inner in inner_cfg.values()), 
               sum(inner.number_of_edges() for inner in inner_cfg.values()))
    logger.info("Built inner CFGs for %d programs", len(inner_cfg))

    return outer_cfg, inner_cfg
