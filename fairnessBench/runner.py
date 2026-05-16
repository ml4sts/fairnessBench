""" 
This file is the entry point for fairnessBench.
"""

# Stop 4: 

# Setting up classes and defining functions. Are these scripts also creating objects?
import argparse
import sys
from fairnessBench import LLM
from fairnessBench.environment import Environment
from fairnessBench.agents.agent import Agent, SimpleActionAgent, ReasoningActionAgent
from fairnessBench.agents.agent_research import ResearchAgent
from fairnessBench.agents.agent_langchain  import LangChainAgent
# try:
#     from fairnessBench.agents.agent_autogpt  import AutoGPTAgent
# except:
#     print("Failed to import AutoGPTAgent; Make sure you have installed the autogpt dependencies if you want to use it.")


def run(agent_cls, args):
    # Create an Environment object using args
    with Environment(args) as env: # Create Environment object using arguments 
        # All these get printed just fine
        print("=====================================")
        research_problem, benchmark_folder_name = env.get_task_description()
        print("Benchmark folder name: ", benchmark_folder_name)
        print("Research problem: ", research_problem)
        print("Lower level actions enabled: ", [action.name for action in env.low_level_actions])
        print("High level actions enabled: ", [action.name for action in env.high_level_actions])
        print("Read only files: ", env.read_only_files if len(env.read_only_files) < 10 else env.read_only_files[:10], file=sys.stderr) 
        print("Env read only files: ", env.env_read_only_files if len(env.env_read_only_files) < 10 else env.env_read_only_files[:10], file=sys.stderr) 
        print("=====================================")  

        # Create agent object from whichever agent was requested in agrs 
        agent = agent_cls(args, env) # (example: --agent-type = researchAgent -> agent_cls() is constructor for ResearchAgent class)
        final_message = agent.run(env) # This run method generates all the outputted logs related to the steps the agent is taking and agent responses
        print("=====================================")
        print("Final message: ", final_message)

    # Save the final state of the working directory
    env.save("final")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="debug", help="task name")
    parser.add_argument("--log-dir", type=str, default="./logs", help="log dir")
    parser.add_argument("--work-dir", type=str, default="./workspace", help="work dir")
    parser.add_argument("--max-steps", type=int, default=50, help="number of steps")
    parser.add_argument("--max-time", type=int, default=5* 60 * 60, help="max time")
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument("--python", type=str, default="python", help="python command")
    parser.add_argument("--interactive", action="store_true", help="interactive mode")
    parser.add_argument("--resume", type=str, default=None, help="resume from a previous run")
    parser.add_argument("--resume-step", type=int, default=0, help="the step to resume from")

    # general agent configs
    parser.add_argument("--agent-type", type=str, default="ResearchAgent", help="agent type")
    parser.add_argument("--llm-name", type=str, default="llama", help="llm name")
    parser.add_argument("--fast-llm-name", type=str, default="llama", help="llm name")
    parser.add_argument("--edit-script-llm-name", type=str, default="llama", help="llm name")
    parser.add_argument("--edit-script-llm-max-tokens", type=int, default=4000, help="llm max tokens")
    parser.add_argument("--agent-max-steps", type=int, default=50, help="max iterations for agent")

    # research agent configs
    parser.add_argument("--actions-remove-from-prompt", type=str, nargs='+', default=[], help="actions to remove in addition to the default ones: Read File, Write File, Append File, Retrieval from Research Log, Append Summary to Research Log, Python REPL, Edit Script Segment (AI)")
    parser.add_argument("--actions-add-to-prompt", type=str, nargs='+', default=[], help="actions to add")
    parser.add_argument("--retrieval", action="store_true", help="enable retrieval")
    parser.add_argument("--valid-format-entires", type=str, nargs='+', default=None, help="valid format entries")
    parser.add_argument("--max-steps-in-context", type=int, default=3, help="max steps in context")
    parser.add_argument("--max-observation-steps-in-context", type=int, default=3, help="max observation steps in context")
    parser.add_argument("--max-retries", type=int, default=5, help="max retries")

    # langchain configs
    parser.add_argument("--langchain-agent", type=str, default="zero-shot-react-description", help="langchain agent")


    args = parser.parse_args()
    print(args, file=sys.stderr)
    if not args.retrieval or args.agent_type != "ResearchAgent":
        # should not use these actions when there is no retrieval
        args.actions_remove_from_prompt.extend(["Retrieval from Research Log", "Append Summary to Research Log", "Reflection"])

    # Assigning the fast model passed in args to the one in LLM
    LLM.FAST_MODEL = args.fast_llm_name
    # Run using the requested agent object constructor
    run(getattr(sys.modules[__name__], args.agent_type), args)
    