To get the evaluation running you need to setup a conda environment and the environment variables, addtionally you need fast downward.
Furthermore, you will need an OpenAI API key.
The packages were tested on Ubuntu 18 and 22.

First, open your console in this directory.

```
git clone https://github.com/aibasel/downward.git

# environment variables (could also be added to .bashrc) , don't forget to change the OPENAI API KEY!
export FAST_DOWNWARD_ROOT=$(pwd)/downward
export AUTOGPT_ROOT=$(pwd)/autogpt_p
export OAM_ROOT=$(pwd)/object_affordance_mapping
export OPENAI_API_KEY='sk-your-key'


# create environment and install autogpt+p and dependencies
conda create --name autogpt-p python=3.8
conda activate autogpt-p
cd downward/
python build.py
cd ..
cd autogpt_p/
pip install -e .

# go to the evaluation directory and execute one of the evaluations
# if you want to change the configuration, it is best to open autogpt_p with an IDE, and change the config-object at the bottom of the file that ends with _evaluation.py, i.e. planner_evaluation.py
# Warning: Running this appends the results to the original logs, which may be confusing. It may be useful to copy them first.
cd autogpt_p/evaluation/
python alternative_suggestion_evaluation.py
python planner_evaluation.py
python autogpt_p_evaluation.py
```
