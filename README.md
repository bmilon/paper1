<!--
README Documentation

This README provides an overview of the POMS 2025 submission repository. It outlines the repository's contents, including source code, data files, and documentation. The requirements section specifies the need for Python 3.x and additional packages listed in `requirements.txt`. Usage instructions guide users through cloning the repository, installing dependencies, and running the main script. Licensing information and contact details for the project maintainer are also included.
-->

## Overview
This repository contains the submission materials for our paper, titled "Optimizing Delivery for Quick Commerce Factoring Qualitative Assessment of Generated Routes." The project presents a framework for optimizing delivery routes in quick commerce scenarios by integrating both quantitative metrics and qualitative assessments of generated routes. The repository includes Python source code for route generation, data files for experimentation, and scripts to evaluate route quality using large language models (LLMs). The approach enables users to sample and analyze delivery routes, automate qualitative feedback extraction, and benchmark different routing strategies. Detailed instructions and requirements are provided to facilitate reproducibility and further research. The novelty of our approach is that 
* Conventionally, routing (done using VRP/CVRP) solution optimize using _crow-flying_ distances to optimize for cost and execution speed. This may lead to instances where the geenrated route might be infeasible.
* The routing tool optimize for either distance travelled or time taken. They dont consider the _quality_ of the route. 
* anecdotal evidence suggests that delivery time also depends either on weather or the landmarks encountered during the delivery process.
* This study is the first step in quantifying the risk associated with such landmarks, where we try to identify such landmarks using large language models (LLMs).  
* Currently, such an evaluation is either performed manually or is dependent of the rider's experience/knowledge. 

## Contents

- Source code
- Data files
- Documentation

## Requirements

- Python 3.x
- Required packages listed in `requirements.txt`

## Usage

1. Clone the repository.
2. Install dependencies:  
    ```
    pip install -r requirements.txt
    ```
3. Run the main script:  
    ```
    python3 main.py --filepath bangalore.shp --drop_points 6 --routes_to_sample  1 --base_url http://localhost:11434/v1/
    ```

## TODO

- Use a reasoning model to extract LLM opinion as a atructured output: <span style="color: green;">Done</span>
- evaluate on bigger models online (e.g. Qwen2.5-VL-32B)
- test on OpenAI models <span style="color: green;">Done</span>
- Add detailed documentation for each module
- Include example input and output files
- Expand test coverage
- Benchmark mistral and GPT-OSS against GPT 5
- Generate a dataset of 400 examples 
- Decide what resolution works best 
- Add examples from different map providers (including Google Maps)
- Describe the dataset in a section in the report 
- Introduce pause to limit overheating of GPU
- Search lakes on OSM and save image to create sample 

## License

See [LICENSE](LICENSE) for details.

## Contact

For questions, please contact  [Milon Bhattacharya](milon.bhattacharya25-08@iimv.ac.in)

