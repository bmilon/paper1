# main.py
# Entry point for the POMS2025 study 

import sys, argparse , os , string , random
from library.generate_training_data import create_and_save_routes
from library.eval_route_LLM import eval_routes_using_LLM, convert_raw_response_to_summary, eval_LLM_answer_against_labelled_data
import geopandas as gpd
import pandas as pd
from tabulate import tabulate




def main(args):

    print(f'arguments used : {args}')

    os.environ['SESSION_KEY'] = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(args.session_key_size))

    if args.use_open_ai:
        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        if not OPENAI_API_KEY:
            print('OPENAI_API_KEY is not set in the environment variables. Please set it.')
            return

        os.environ['key'] =  os.environ['OPENAI_API_KEY']
        args.llms_to_use= ["gpt-5"]       
    else:
        os.environ['key']='ollama' 

    # Step 1: generate data if not already done
    print("Starting simulation")

    if args.generate_fresh_data:
        print('Generating new routes for testing ')
        create_and_save_routes(args)
        try:
            ...
            #create_and_save_routes(args)
        except Exception as e:
            print(f"Error occurred while creating routes, stopping the simulation : {e}")
            return  # Stop the simulation if route creation fails
    
    if args.evaluate_files:
        # Step 2: performing LLM based evaluation
        print(f'Evaluating generated routes in folder: {args.path_to_save_individual_legs}')
        print(f'Using models:{args.llms_to_use} for evaluation ')

        eval_df = eval_routes_using_LLM(
            eval_folder=args.path_to_save_individual_legs,
            sim_count=max(args.routes_to_sample*(args.drop_points+1),args.sim_count),
            llms_to_use=args.llms_to_use,
            use_pydantic=args.use_pydantic,
            max_attempts=args.max_attempts,
            temp_incr=args.temp_incr,
            args=args
        )
        #print(tabulate(eval_df, headers='keys', tablefmt='psql'))
        print(eval_df.head())
        current_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        if args.extract_structured_output:
            # Step 3.1: summarize response of all raw type into single output
            print('attempting to extraction structured output from the raw LLM response.')
            
            formatted_response_df = \
                pd.DataFrame([convert_raw_response_to_summary(raw_response=r,base_url=args.base_url,key=os.environ['key'], \
                                                            model=args.model_for_struct_output) for r in eval_df['raw_response']])

            try:
                eval_df = pd.concat([eval_df, formatted_response_df], axis=1)
            except Exception as e:
                print(f"Error occurred while summarizing responses: {e}, writing only raw responses to CSV.")

        # Step 3.2: save evaluation results
        os.makedirs(args.path_to_save_evaluations, exist_ok=True)
        args.path_to_predictions_data_csv =\
              os.path.join(args.path_to_save_evaluations + f"eval_results_{os.environ['SESSION_KEY']}_{args.routes_to_sample}_{current_date}.csv")
        
        if len(eval_df) > 0:
            print(f'saving LLM outputs at :{args.path_to_predictions_data_csv}')
            eval_df.to_csv(args.path_to_predictions_data_csv, index=False)
        else:
            print('No data to save eval_df is empty.')
        
    if args.compare_with_labelled_data: # Step 4:           
        llm_eval_results_path = os.path.join(args.path_to_save_evaluations + \
                                                f"llm_eval_results_for_{args.path_to_predictions_data_csv.split('/')[-1].split('.')[0]}_llm_count_{len(args.llms_to_use)}_.csv")

        llm_eval_results = eval_LLM_answer_against_labelled_data(path_to_labeled_data_json=args.path_to_labeled_data,\
                                                path_to_predictions_data_csv=args.path_to_predictions_data_csv,args=args,eval_model=args.model_for_struct_output,)
        print(f'saving evaluation results at :{llm_eval_results_path}')
        llm_eval_results.to_csv(llm_eval_results_path, index=True) # filenames are index 
        try:
            ...
            if llm_eval_results:
                print(f'saving evaluation results at :{llm_eval_results_path}')                
        except:
            print('cannot generate evaluation results')
    # thanks you !
    print('Simulation completed! ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple program demonstrating argparse.")
    parser.add_argument("--filepath", help="Path to the shapefile")
    parser.add_argument("--drop_points", type=int, default=5, help="Number of points to sample")
    parser.add_argument("--session_key_size", type=int, default=5, help="Size of session key")
    parser.add_argument("--max_timer", type=int, default=10, help="parameter which controls maximum pause b/w each LLM call ")
    parser.add_argument("--sim_count", type=int, default=1, help="Number of generated files to evaluate")
    parser.add_argument("--routes_to_sample", type=int, default=5, help="Number of simulations to run")
    parser.add_argument("--graph_file_path", help="Path to the graph file",default=r'training_data/study_area/blr.graphml')
    parser.add_argument("--path_to_save_route_order", help="Path to save the routing order", default=r"training_data/routing/")
    parser.add_argument("--path_to_save_directions", help="Path to save the driving directions", default=r"training_data/driving_directions/")
    parser.add_argument("--path_to_save_individual_legs", help="Path to save the individual legs of the route", default=r"training_data/individual_legs/")
    parser.add_argument("--path_to_save_evaluations", help="Path to save the evaluation results", default=r"training_data/results/")
    parser.add_argument("--path_to_labeled_data", help="Folder to save the evaluation results (JSON format)", default=None)
    parser.add_argument("--path_to_predictions_data_csv", help="Path to save the evaluation results CSV", default=None)

    parser.add_argument("--llms_to_use", nargs='+', help="List of LLMs to use for this evaluation", default=['qwen2.5vl:7b','gemma3:12b','mistral-small3.2:latest','llama3.2-vision:latest','minicpm-v:latest','granite3.2-vision:latest',])
    parser.add_argument("--temperature", help="Temperature to use for the LLMs", default=0.0)
    parser.add_argument("--base_url", help="Base URL for the LLMs", default='http://localhost:11434/v1/')
    parser.add_argument("--max_attempts", type=int, default=3, help="Maximum number of attempts for LLM calls")
    parser.add_argument("--temp_incr", type=float, default=0.05, help="Temperature increment for retries")
    parser.add_argument("--use_pydantic", action="store_true", help="Whether to use Pydantic for response validation")
    parser.add_argument("--line_strings_to_sample", type=int, default=1, help="Number of line strings to sample")
    parser.add_argument("--model_for_struct_output", type=str, default="mistral-small3.2:latest", help="Model to be used to extract structured JSON from the LLM output")
    parser.add_argument("--use_open_ai", default=False, action="store_true", help="if use GPT-5 from OpenAI for inference (also modify llms_to_use)")    
    parser.add_argument("--add_pause_bw_evals", default=False, action="store_true", help="add a pause between each evaluation so that GPU stays below critical load (prevent overheating)")    
    parser.add_argument("--generate_fresh_data", default=False,  action="store_true", help="Set this flag to generate new data and save them in path_to_save_individual_legs")    
    parser.add_argument("--evaluate_files", action="store_true", help="Set this flag to evaluate images in path_to_save_individual_legs")
    parser.add_argument("--compare_with_labelled_data", action="store_true", help="Set this flag to evaluate images in path_to_save_evaluations and path to labelled data")            
    parser.add_argument("--extract_structured_output", action="store_true", help="Set this flag to evaluate extract JSON output from LLM output (experimental)")        
    parser.add_argument("--dont_save_progress",default=False, action="store_false", help="This parameter is used to control checkpointing of evaluation results")        
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
