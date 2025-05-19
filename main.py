import pickle
import os
import logging
from graph_builder import build_graphs
import asyncio
# Import the new reverse documentation workflow instead
from reverse_documentation_workflow import compiled_workflow, initialize_reverse_workflow
import json
import glob

# Configure logging for the entire application
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('application.log', mode='w'),  # Overwrite for each run
        logging.StreamHandler()
    ]
)

GRAPH_CACHE_FILE = "graph_cache.pkl"
TEST_DATA_DIR = "test_data"

def load_json_files_from_dir(directory_path, file_pattern):
    """
    Loads and parses JSON files matching a pattern from a directory.
    """
    json_data_list = []
    search_path = os.path.join(directory_path, file_pattern)
    for file_path in glob.glob(search_path):
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                json_data_list.append(json_data)
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {file_path}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while reading {file_path}: {e}")
    return json_data_list


if __name__ == "__main__":
    # Load data from JSON files in test_data directory
    logging.info(f"Loading data from {TEST_DATA_DIR}...")
    jcl_jsons = load_json_files_from_dir(os.path.join(TEST_DATA_DIR, "jcl"), "*.json")
    cobol_jsons = load_json_files_from_dir(os.path.join(TEST_DATA_DIR, "cobol"), "*/*.json")

    if not jcl_jsons and not cobol_jsons:
        logging.error(f"No JSON files found in {TEST_DATA_DIR} matching patterns jcl_*.json or cobol_*.json. Exiting.")
        exit()

    outer_cfg = None
    inner_cfgs = None

    outer_cfg, inner_cfgs = build_graphs(jcl_jsons, cobol_jsons)
    exit()
    # Try to load graphs from cache
    if os.path.exists(GRAPH_CACHE_FILE):
        logging.info(f"Loading graphs from cache: {GRAPH_CACHE_FILE}")
        try:
            with open(GRAPH_CACHE_FILE, 'rb') as f:
                outer_cfg, inner_cfgs = pickle.load(f)
            logging.info("Graphs loaded successfully from cache.")
        except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
            logging.warning(f"Could not load graphs from cache: {e}. Rebuilding...")
            outer_cfg = None # Ensure we rebuild if loading fails
            inner_cfgs = None
    
    if outer_cfg is None or inner_cfgs is None:
        # Build graphs if not loaded from cache or if loading failed
        logging.info("Building graphs...")
        outer_cfg, inner_cfgs = build_graphs(jcl_jsons, cobol_jsons)
        logging.info("Graphs built successfully.")
        # Save graphs to cache
        try:
            with open(GRAPH_CACHE_FILE, 'wb') as f:
                pickle.dump((outer_cfg, inner_cfgs), f)
            logging.info(f"Graphs saved to cache: {GRAPH_CACHE_FILE}")
        except Exception as e:
            logging.error(f"Could not save graphs to cache: {e}")
    # Run LangGraph-based reverse documentation workflow
    if outer_cfg and inner_cfgs:
        logging.info("Starting reverse documentation workflow...")
        output_directory = "generated_documentation"
        # Initialize state using the reverse workflow function
        initial_state = initialize_reverse_workflow(outer_cfg, inner_cfgs, output_directory)
        # The workflow is already compiled in the imported module
        # Run the workflow
        result = asyncio.run(compiled_workflow.ainvoke(initial_state, config = {"recursion_limit": 5000}))
        logging.info("Reverse documentation workflow finished successfully.")
    else:
        logging.error("Graphs could not be built or loaded. Skipping documentation workflow.")


