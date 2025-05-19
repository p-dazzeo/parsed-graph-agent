import pickle
import os
import logging
from graph_builder import build_graphs
import asyncio
from documentation_workflow import workflow, initialize_state

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

if __name__ == "__main__":
    # sample_jcl.json
    sample_jcl = {
        "jobName": "DAILY_JOB",
        "stepNumber": 1,
        "step": {
            "stepName": "STEP01",
            "programId": "PROGRAM_A",
            "procName": None,
            "isProc": False,
            "datasets": [
                {
                    "ddname": "INFILE",
                    "datasetName": "CUSTOMER.DATA",
                    "disp": "SHR"
                }
            ],
            "codeWithComments": "//STEP01 EXEC PGM=PROGRAM_A\n//INFILE DD DSN=CUSTOMER.DATA,DISP=SHR",
            "codeWithoutComments": "EXEC PGM=PROGRAM_A\nINFILE DD DSN=CUSTOMER.DATA,DISP=SHR"
        }
    }

    # sample_program_a.json
    sample_program_a = {
        "identification_division": {
            "program_id": "PROGRAM_A"
        },
        "environment_division": {},
        "data_division": {
            "file_section": {},
            "working_storage_section": {
                "vars": ["WS-CUST-ID", "WS-NAME"]
            },
            "linkage_section": {}
        },
        "procedure_division_using": [],
        "paragraph": {
            "INIT": {
                "called_programs": [],
                "perform_targets": ["PROCESS"],
                "goto_targets": [],
                "codeWithComments": "* INIT: load input file\nMOVE 1 TO WS-CUST-ID\nPERFORM PROCESS",
                "codeWithoutComments": "MOVE 1 TO WS-CUST-ID\nPERFORM PROCESS"
            },
            "PROCESS": {
                "called_programs": ["SUBPROGRAM_B"],
                "perform_targets": [],
                "goto_targets": ["END"],
                "codeWithComments": "* PROCESS: call SUBPROGRAM_B\nCALL 'SUBPROGRAM_B' USING WS-CUST-ID\nGO TO END",
                "codeWithoutComments": "CALL 'SUBPROGRAM_B' USING WS-CUST-ID\nGO TO END"
            },
            "END": {
                "called_programs": [],
                "perform_targets": [],
                "goto_targets": [],
                "codeWithComments": "* END of program",
                "codeWithoutComments": "STOP RUN"
            }
        }
    }
    # sample_subprogram_b.json
    sample_program_b = {
        "identification_division": {
            "program_id": "SUBPROGRAM_B"
        },
        "environment_division": {},
        "data_division": {
            "file_section": {},
            "working_storage_section": {},
            "linkage_section": {
                "vars": ["WS-CUST-ID"]
            }
        },
        "procedure_division_using": ["WS-CUST-ID"],
        "paragraph": {
            "VALIDATE": {
                "called_programs": [],
                "perform_targets": [],
                "goto_targets": [],
                "codeWithComments": "* VALIDATE: check ID\nIF WS-CUST-ID > 0\nDISPLAY 'Valid ID'\nEND-IF",
                "codeWithoutComments": "IF WS-CUST-ID > 0\nDISPLAY 'Valid ID'\nEND-IF"
            }
        }
    }

    # Load mock data
    jcl_jsons = [sample_jcl]
    cobol_jsons = [sample_program_a, sample_program_b]

    outer_cfg = None
    inner_cfgs = None

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


    # Run LangGraph-based documentation workflow
    if outer_cfg and inner_cfgs:
        logging.info("Starting LangGraph documentation workflow...")
        output_directory = "generated_documentation"
        initial_state = initialize_state(outer_cfg, inner_cfgs, output_directory)
        # Compile the workflow (can be done once outside if preferred)
        app = workflow.compile()
        # Invoke the workflow
        # LangGraph invoke is synchronous by default, use ainvoke for async
        # For simplicity here, we'll use invoke for now, assuming nodes are sync
        # Invoke the workflow asynchronously
        result = asyncio.run(app.ainvoke(initial_state))
        logging.info("LangGraph documentation workflow finished.")
        # The result contains the final state, including documented_nodes
        # You might want to add logic here to process/save the final result
        # For example, saving the documented_nodes dictionary
        # save_documentation_to_files(result["documented_nodes"], output_directory)
    else:
        logging.error("Graphs could not be built or loaded. Skipping documentation workflow.")
