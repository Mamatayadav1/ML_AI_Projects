import random
from llama_cpp import Llama

# ---------------------------
# Destinations Plugin
# ---------------------------
class DestinationsPlugin:
    """A List of Random Destinations for a vacation."""

    def __init__(self):
        self.destinations = [
            "Barcelona, Spain",
            "Paris, France",
            "Berlin, Germany",
            "Tokyo, Japan",
            "Sydney, Australia",
            "New York, USA",
            "Cairo, Egypt",
            "Cape Town, South Africa",
            "Rio de Janeiro, Brazil",
            "Bali, Indonesia",
            "Delhi, India"
        ]
        self.last_destination = None

    def get_random_destination(self) -> str:
        available_destinations = self.destinations.copy()
        if self.last_destination and len(available_destinations) > 1:
            available_destinations.remove(self.last_destination)
        destination = random.choice(available_destinations)
        self.last_destination = destination
        return destination

# ---------------------------
# Local GPT4All (Qwen2) model
# ---------------------------
MODEL_PATH = r"qwen2-1_5b-instruct-q4_0.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=6
)

def local_completion(prompt: str, max_tokens: int = 200) -> str:
    result = llm(prompt, max_tokens=max_tokens, stop=["</s>"])
    return result['choices'][0]['text']

# ---------------------------
# Offline Agent
# ---------------------------
class OfflineTravelAgent:
    def __init__(self):
        self.plugin = DestinationsPlugin()
        self.conversation_history = "You are a helpful travel assistant.\n"

    def respond(self, user_input: str) -> str:
        # If the user asks for a destination
        if "destination" in user_input.lower() or "trip" in user_input.lower():
            dest = self.plugin.get_random_destination()
            prompt = f"Plan a day trip to {dest}."
        else:
            prompt = user_input

        # Add to conversation history
        self.conversation_history += f"User: {user_input}\nAI: "
        response = local_completion(self.conversation_history + prompt)
        self.conversation_history += response + "\n"
        return response

# ---------------------------
# Interactive Loop
# ---------------------------
if __name__ == "__main__":
    agent = OfflineTravelAgent()
    print("Offline Travel Agent (type 'exit' to quit)")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = agent.respond(user_input)
        print(f"TravelAgent: {response}")
