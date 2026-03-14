"""Example usage of Z.AI API Client."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from zai.client import ZAIClient
from zai.core.exceptions import ZAIError


def main():
    """
    Main function demonstrating Z.AI client usage.
    """
    try:
        print("Initializing Z.AI Client (JAEGIS Nexus Core)...")
        client = ZAIClient(auto_auth=True, verbose=False)
        print(f"Got token: {client.token[:20]}...")
        
        # ==============================================================================
        # EXAMPLE 1: GLM-4.7 - Advanced Tooling & Deep Research
        # ==============================================================================
        print("\n" + "="*50)
        print("Test 1: GLM-4.7 (Full-Stack, AI Slides, Magic Design, Deep Research)")
        print("="*50)
        
        try:
            response_47 = client.simple_chat(
                message="Research the latest UI trends and generate a full-stack dashboard layout with accompanying AI slides.",
                model="glm-4.7",
                system_prompt="You are an expert Full-Stack AI developer and researcher utilizing the JAEGIS Cognitive Loom.",
                enable_thinking=True,
                web_search=True,
                # New GLM-4.7 Feature Toggles:
                full_stack=True,
                ai_slides=True,
                magic_design=True,
                deep_research=True,
                # Standard Params:
                temperature=0.7,
                top_p=0.9,
                max_tokens=2000
            )
            
            if response_47.content:
                print(f"\nResponse (GLM-4.7): {response_47.content[:500]}...\n[Content Truncated]")
            if response_47.thinking:
                print(f"\nThinking: {response_47.thinking[:200]}...")
                
        except Exception as chat_error:
            print(f"GLM-4.7 Chat error: {chat_error}")

        # ==============================================================================
        # EXAMPLE 2: GLM-5 - Autonomous Agent Mode
        # ==============================================================================
        print("\n" + "="*50)
        print("Test 2: GLM-5 (Agent Mode / Meta-Training & Orchestration)")
        print("="*50)
        
        try:
            response_5 = client.simple_chat(
                message="Autonomously orchestrate a data analysis pipeline for the attached metrics and resolve any conflicts using the Problem Meets Problem Protocol.",
                model="glm-5",
                system_prompt="You are the primary orchestrator agent running on GLM-5.",
                enable_thinking=True,
                web_search=True,
                # New GLM-5 Feature Toggle:
                agent=True,
                # Standard Params:
                temperature=0.5,
                top_p=0.95,
                max_tokens=2000
            )
            
            if response_5.content:
                print(f"\nResponse (GLM-5): {response_5.content[:500]}...\n[Content Truncated]")
            if response_5.usage:
                print(f"\nUsage: {response_5.usage}")
                
        except Exception as chat_error:
            print(f"GLM-5 Chat error: {chat_error}")
        
    except ZAIError as e:
        print(f"ZAI Error: {e}")
    except Exception as e:
        import traceback
        print(f"Unexpected error: {e}")
        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
