from crewai import LLM

# LLM instance configured to use local Ollama (Mistral)
llm = LLM(
    model="ollama/mistral",             
    base_url="http://localhost:11434"   # Default Ollama API
)
