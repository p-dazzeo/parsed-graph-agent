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
    for prog in cobol_json_list:
        prog_id = prog['identification_division']['program_id']
        logger.info("Processing COBOL program: %s", prog_id)
        # ensure node exists in outer graph
        outer_cfg.add_node(prog_id, type='program',
                           identification_division=prog.get('identification_division', {}),
                           environment_division=prog.get('environment_division', {}),
                           data_division=prog.get('data_division', {}),
                           procedure_division_using=prog.get('procedure_division_using', []),
                           has_inner_cfg=bool(prog.get('paragraph')))
        
        # Collect code snippets from paragraphs for the outer graph node
        program_code_with_comments = ""
        program_code_without_comments = ""
        paragraphs = prog.get('paragraph', {})
        for p_name, p_details in paragraphs.items():
            program_code_with_comments += f"\n* {p_name}:\n{p_details.get('codeWithComments', '')}"
            program_code_without_comments += f"\n{p_details.get('codeWithoutComments', '')}"

        outer_cfg._graph.nodes[prog_id]['codeWithComments'] = program_code_with_comments.strip()
        outer_cfg._graph.nodes[prog_id]['codeWithoutComments'] = program_code_without_comments.strip()
        # build inner CFG for this program
        inner = Graph()        # assume 'paragraphs' is a list of paragraph dicts
        paragraphs = prog.get('paragraph')
        logger.debug("Found %d paragraphs in program %s", len(paragraphs), prog_id)
        # add nodes
        for p,v in paragraphs.items():
            pid = f"{prog_id}:{p}"
            inner.add_node(pid,
                           name=p,
                           codeWithComments=v['codeWithComments'],
                           codeWithoutComments=v['codeWithoutComments'],
                           type='paragraph')
        # add edges for PERFORM targets
        for p,v in paragraphs.items():
            src = f"{prog_id}:{p}"
            for tgt in v.get('perform_targets', []):
                tgt_id = f"{prog_id}:{tgt}"
                inner.add_edge(src, tgt_id, type='PERFORM')
            for tgt in v.get('goto_targets', []):
                tgt_id = f"{prog_id}:{tgt}"
                inner.add_edge(src, tgt_id, type='GOTO')        # remove dead code (no incoming except entry)
        entry = f"{prog_id}:ENTRY"
        dead_nodes = []
        for node in list(inner.nodes()):
            if node != entry and inner.in_degree(node) == 0:
                dead_nodes.append(node)
                inner.remove_node(node)
        if dead_nodes:
            logger.debug("Removed %d dead nodes from program %s: %s", 
                        len(dead_nodes), prog_id, dead_nodes)
        
        # break cycles: remove back-edges for i<j by name order
        removed_edges = []
        for u, v in list(inner.edges()):
            if inner.has_path(v, u):
                # remove edge v->u to break cycle
                removed_edges.append((v, u))
                inner.remove_edge(v, u)
        if removed_edges:
            logger.debug("Removed %d edges to break cycles in program %s", 
                        len(removed_edges), prog_id)
        
        inner_cfg[prog_id] = inner        # inter-program CALL edges
        called = prog['paragraph'].get('called_programs', [])
        if called:
            logger.debug("Program %s calls %d other programs: %s", 
                        prog_id, len(called), called)
        for callee in called:
            outer_cfg.add_node(callee, type='program')
            outer_cfg.add_edge(prog_id, callee, type='CALL')

    # --- Process JCL jobs ---
    for job in jcl_json_list:
        job_name = job['jobName']
        step = job['step']
        step_id = f"{job_name}:{step['stepName']}"
        logger.info("Processing JCL job step: %s", step_id)
        # JCL step node
        outer_cfg.add_node(step_id, type='jcl_step',
                           datasets=step.get('datasets', []),
                           codeWithComments=step.get('codeWithComments', ''),
                           codeWithoutComments=step.get('codeWithoutComments', ''))
        # link JCL->COBOL        prog_id = step['programId']
        outer_cfg.add_node(prog_id, type='program')
        outer_cfg.add_edge(step_id, prog_id, type='EXECUTES')

    logger.info("Graph building completed. Outer graph has %d nodes and %d edges.", 
               outer_cfg.number_of_nodes(), outer_cfg.number_of_edges())
    logger.info("Built inner CFGs for %d programs", len(inner_cfg))
    
    return outer_cfg, inner_cfg
