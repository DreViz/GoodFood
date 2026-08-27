def mock_planner_output(user_text: str):
    """
    Fallback response when Ollama is not running or planner fails.
    """
    return {
        "plan": "reply",
        "reply": "Hi there! I'm GoodFoods Concierge. How can I help you with your dining plans today?"
    }
