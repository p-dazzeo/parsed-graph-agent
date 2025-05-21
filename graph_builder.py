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

def debug_print_graph(graph, filename=None, use_interactive=True):
    """
    Print a visual representation of the graph. Optionally save to a file.
    
    Args:
        graph (Graph): The graph to visualize.
        filename (str, optional): Path to save the visualization. If None, just displays.
        use_interactive (bool): Whether to use interactive visualization (HTML) or static (PNG).
    """
    import os
    
    # Create output directory if it doesn't exist
    output_dir = "output/images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Update filename with output directory if provided
    if filename:
        filename = os.path.join(output_dir, os.path.basename(filename))
    
    if use_interactive:
        try:
            from pyvis.network import Network
            import networkx as nx
            import webbrowser
            
            # Create PyVis network
            net = Network(height="750px", width="100%", notebook=False, directed=True)
            
            # From NetworkX graph to PyVis network
            net.from_nx(graph._graph)
            
            # Configure physics for better visualization
            net.toggle_physics(True)
            net.set_options("""
            const options = {
                "nodes": {
                    "font": {
                        "size": 12,
                        "face": "Tahoma"
                    },
                    "shape": "dot",
                    "size": 15
                },
                "edges": {
                    "color": {
                        "inherit": true
                    },
                    "smooth": {
                        "enabled": false
                    },
                    "arrows": {
                        "to": {
                            "enabled": true,
                            "scaleFactor": 0.5
                        }
                    }
                },
                "physics": {
                    "barnesHut": {
                        "gravitationalConstant": -4000,
                        "centralGravity": 0.3,
                        "springLength": 120
                    },
                    "minVelocity": 0.75
                }
            }""")
            
            # Save and show
            html_file = filename.replace('.png', '.html') if filename else os.path.join(output_dir, "network.html")
            net.save_graph(html_file)
            print(f"Interactive graph saved to {html_file}")
            
            # Open in browser
            try:
                webbrowser.open('file://' + os.path.abspath(html_file), new=2)
            except Exception as e:
                print(f"Couldn't open browser automatically: {e}\nPlease open {html_file} manually.")
            
        except ImportError:
            print("PyVis not installed. Falling back to static visualization.")
            print("To install: pip install pyvis")
            _static_graph_visualization(graph, filename)
    else:
        _static_graph_visualization(graph, filename)


def _static_graph_visualization(graph, filename=None):
    """
    Create a static graph visualization using matplotlib.
    
    Args:
        graph (Graph): The graph to visualize.
        filename (str, optional): Path to save the visualization.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import os
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph._graph, seed=42)  # For reproducible layout
    nx.draw(graph._graph, pos, with_labels=True, node_color='lightblue', 
            node_size=500, arrows=True, connectionstyle='arc3,rad=0.1')
    
    if filename:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    plt.show()


def debug_print_graph_dict(graph_dict, base_filename=None, use_interactive=True):
    """
    Print visual representations of a dictionary of graphs.
    Optionally save each to a file with program ID in the filename.
    
    Args:
        graph_dict (dict): Dictionary mapping program IDs to Graph objects.
        base_filename (str, optional): Base path for saving visualizations.
        use_interactive (bool): Whether to use interactive visualization (HTML) or static (PNG).
    """
    import os
    
    # The output directory is handled within debug_print_graph
    # but we need to sanitize program IDs for filenames    
    for prog_id, graph in graph_dict.items():
        print(f"Visualizing graph for program: {prog_id}")
        if base_filename:
            # Handle different file extensions for interactive vs static
            ext = "html" if use_interactive else base_filename.split('.')[-1]
            base = base_filename.split('.')[0]
            
            # Sanitize program_id for use in filename (remove special chars)
            safe_prog_id = "".join(c if c.isalnum() else "_" for c in prog_id)
            filename = f"{base}_{safe_prog_id}.{ext}"
            
            debug_print_graph(graph, filename, use_interactive)
        else:
            debug_print_graph(graph, None, use_interactive)


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
        prog_id = paragraph_json['identification_division']['program_id']
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
        remove_edges(inner, list(inner.edges()))
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
    
    # If you want to visualize graphs, uncomment the following lines:
    # # Visualize the outer graph (interactive HTML)
    # debug_print_graph(outer_cfg, 'outer_cfg.png')
    # 
    # # Visualize inner graphs for each program (interactive HTML)
    debug_print_graph_dict(inner_cfg, 'inner_cfg.png')
    #
    # # For static PNG visualizations instead, use:
    # # debug_print_graph(outer_cfg, 'outer_cfg.png', use_interactive=False)
    # # debug_print_graph_dict(inner_cfg, 'inner_cfg.png', use_interactive=False)

    return outer_cfg, inner_cfg


def visualize_graphs(outer_cfg=None, inner_cfg=None, interactive=True):
    """
    Visualize the graphs generated by build_graphs.
    
    Args:
        outer_cfg (Graph, optional): The outer graph to visualize.
        inner_cfg (dict, optional): Dictionary mapping program IDs to inner graphs.
        interactive (bool): Whether to use interactive HTML visualization (True) or static PNG (False).
    """
    import os
    
    # Ensure output directory exists
    output_dir = "output/images"
    os.makedirs(output_dir, exist_ok=True)
    
    if outer_cfg is not None:
        print("Visualizing outer graph...")
        debug_print_graph(outer_cfg, 'outer_cfg.png', use_interactive=interactive)
        
    if inner_cfg is not None:
        print("Visualizing inner graphs...")
        debug_print_graph_dict(inner_cfg, 'inner_cfg.png', use_interactive=interactive)

# Example usage:
# from graph_builder import build_graphs, visualize_graphs
# outer_cfg, inner_cfg = build_graphs(jcl_json_list, cobol_json_list)
# visualize_graphs(outer_cfg, inner_cfg)  # Interactive HTML
# visualize_graphs(outer_cfg, inner_cfg, interactive=False)  # Static PNG

