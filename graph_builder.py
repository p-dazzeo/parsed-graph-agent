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
                           identification_division=prog_details.get('identification_division', {}),
                           environment_division=prog_details.get('environment_division', {}),
                           data_division=prog_details.get('data_division', {}),
                           procedure_division_using=prog_details.get('procedure_division_using', []),
                           has_inner_cfg=bool(paragraphs))
        
        # Collect code snippets from paragraphs for the outer graph node
        program_code_with_comments = ""
        program_code_without_comments = ""
        all_called_programs = set()
        
        # Sort paragraphs by their order in the source code if 'paragraph_order' is available
        sorted_paragraph_items = sorted(
            paragraphs.items(),
            key=lambda item: item[1].get('procedure_division', {}).get('paragraph', {}).get('paragraph_order', float('inf'))
        )
        
        for p_name, p_details in sorted_paragraph_items:
            proc_div = p_details.get('procedure_division', {})
            paragraph_specific_data = proc_div.get('paragraph', {})
            
            actual_para_code_with_comments = paragraph_specific_data.get('code_with_comments', '')
            actual_para_code_without_comments = paragraph_specific_data.get('code_without_comments', '')
            
            program_code_with_comments += f"\n\n* --- PARAGRAPH: {p_name} ---\n"
            program_code_with_comments += actual_para_code_with_comments
            program_code_without_comments += f"\n{actual_para_code_without_comments}"
            
            # Collect called programs from all paragraphs correctly
            for called_prog in paragraph_specific_data.get('called_programs', []):
                all_called_programs.add(called_prog)

        outer_cfg._graph.nodes[prog_id]['code_with_comments'] = program_code_with_comments.strip()
        outer_cfg._graph.nodes[prog_id]['code_without_comments'] = program_code_without_comments.strip()
        
        # build inner CFG for this program
        inner = Graph()
        logger.debug("Found %d paragraphs in program %s", len(paragraphs), prog_id)
        
        # add nodes
        for p,v in paragraphs.items():
            pid = f"{prog_id}:{p}"
            inner.add_node(pid,
                           name=p,
                           code_with_comments=v['procedure_division']['paragraph'].get('code_with_comments'),
                           code_without_comments=v['procedure_division']['paragraph'].get('code_without_comments'),
                           type='paragraph')
            
        
        # add edges for PERFORM and GOTO targets
        for p,v in paragraphs.items():
            src = f"{prog_id}:{p}"
            para_data = v.get('procedure_division', {}).get('paragraph', {})
            
            for tgt in para_data.get('perform_targets', []):
                tgt_id = f"{prog_id}:{tgt}"
                if tgt_id in inner.nodes(): # Ensure target exists in this program's paragraphs
                    inner.add_edge(src, tgt_id, type='PERFORM')
                else:
                    logger.warning(f"PERFORM target {tgt_id} not found in program {prog_id}")
                    
            for tgt in para_data.get('goto_targets', []):
                tgt_id = f"{prog_id}:{tgt}"
                if tgt_id in inner.nodes(): # Ensure target exists in this program's paragraphs
                    inner.add_edge(src, tgt_id, type='GOTO')
                else:
                     logger.warning(f"GOTO target {tgt_id} not found in program {prog_id}")
        
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
                           jobName=job_name, # Add job name attribute
                           stepName=step.get('stepName'), # Add step name attribute
                           stepNumber=step.get('stepNumber'), # Add step number attribute
                           datasets=step.get('datasets', []),
                           codeWithComments=step.get('codeWithComments', ''),
                           codeWithoutComments=step.get('codeWithoutComments', ''))

        # Link JCL step to its parent JCL job (optional, can rely on attribute)
        # outer_cfg.add_edge(job_name, step_id, type='CONTAINS_STEP') # Example edge

        # link JCL step->COBOL program
        prog_id = step['programId']
        # Ensure program node exists (might already exist from COBOL processing)
        if prog_id not in outer_cfg.nodes():
             outer_cfg.add_node(prog_id, type='program') # Minimal node if not processed as COBOL
        outer_cfg.add_edge(step_id, prog_id, type='EXECUTES')

    logger.info("Graph building completed. Outer graph has %d nodes and %d edges.", 
               outer_cfg.number_of_nodes(), outer_cfg.number_of_edges())
    logger.info("Built inner CFGs for %d programs", len(inner_cfg))

    return outer_cfg, inner_cfg
