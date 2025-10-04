from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
#load from .env
import os
from dotenv import load_dotenv
load_dotenv()



model = OpenAIModel(
    model_name=os.getenv("MODEL_NAME", "agentic-large"),
    provider=OpenAIProvider(
        api_key=os.getenv("API_KEY", ""),
        base_url= os.getenv("BASE_URL", "https://dev-gateway.theagentic.ai/v1"),
    ),
)