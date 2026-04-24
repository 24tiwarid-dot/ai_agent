import os
from dotenv import load_dotenv
from groq import Groq
from agents.tool_researcher import tool_researcher_agent
from agents.comparison_agent import comparison_agent
from agents.guide_writer import guide_writer_agent
from agents.judge_agent import judge_agent
from app import extract_tool_names
import time

def log_step(msg):
    print(f"[LOG] {msg}")

def main():
    load_dotenv()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    category = "AI code assistants"
    model_name = "llama-3.1-8b-instant"
    max_turns = 5
    
    print("Running Agent 1...")
    tool_landscape = tool_researcher_agent(client, category, model_name, max_turns, log_step)
    print("\nLandscape:\n", tool_landscape)
    
    time.sleep(2)
    
    print("\nRunning Agent 2...")
    matrix = comparison_agent(client, tool_landscape, category, model_name, max_turns, log_step)
    print("\nMatrix:\n", matrix)
    
    time.sleep(2)
    
    print("\nRunning Agent 3...")
    guide = guide_writer_agent(client, matrix, category, model_name)
    print("\nGuide:\n", guide[:200], "...")
    
    time.sleep(2)
    
    print("\nRunning Agent 4...")
    tools_list = extract_tool_names(matrix)
    judge_result = judge_agent(client, guide, category, tools_list, model_name, log_step)
    print("\nJudge:\n", judge_result)

if __name__ == "__main__":
    main()
