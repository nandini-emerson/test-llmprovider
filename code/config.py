
import os
from dotenv import load_dotenv

class ConfigError(Exception):
    """Custom exception for missing or invalid configuration."""
    pass

class AgentConfig:
    """
    Configuration management for CRM Quote Creation Agent.
    Handles environment variable loading, API key management, LLM config,
    domain settings, validation, error handling, and defaults.
    """

    # Required environment variables for CRM and RAG
    REQUIRED_ENV_VARS = [
        "CRM_API_ENDPOINT",
        "API_AUTH_TOKEN",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    ]

    # LLM configuration defaults
    LLM_DEFAULTS = {
        "provider": "azure",
        "model": "gpt-4.1-aba",
        "temperature": 0.7,
        "max_tokens": 2000,
        "system_prompt": (
            "You are a professional CRM Quote Creation Agent. Your role is to orchestrate the creation of quotes and quote orders in a CRM system using minimal initial inputs, typically triggered by an email workflow. Follow these instructions:\n\n"
            "1. Validate that customerEmailId is present and syntactically correct, and that receivedDateTime is present and ISO-8601 parseable.\n"
            "2. Use the CRM Contact and Account APIs to resolve the customer and confirm eligibility for quoting.\n"
            "3. If any required data (contact, account, bill-to, ship-to, currency, price list, payment terms, tax region, or line items) is missing, request only the minimum additional information needed to proceed.\n"
            "4. Do not guess or autofill any values. Only proceed with confirmed and validated data.\n"
            "5. Build the quote payload (header and line items) and create the Quote via the CRM API.\n"
            "6. Validate pricing, availability, and discount rules using the appropriate API or rules service.\n"
            "7. Create and submit the Quote Order via the CRM API.\n"
            "8. Ensure idempotency by checking for duplicate requests using requestId/correlationId.\n"
            "9. Return a deterministic, auditable result including quoteId, quoteOrderId, status, timestamps, and any warnings or validation summaries.\n"
            "10. If the process fails, return a concise list of missing or invalid inputs, CRM lookup failures, API errors (with code/message), and the minimal missing inputs required to continue.\n\n"
            "Output format: Always provide a structured response as specified below. If information is not found or cannot be resolved, clearly state what is missing and what is required from the user to proceed."
        ),
        "user_prompt_template": (
            "Please provide the following required information to proceed with quote creation:\n\n"
            "- Customer email address (customerEmailId)\n"
            "- Date and time received (receivedDateTime, ISO-8601 format)\n\n"
            "If you have product/line item details, please include product/SKU, quantity, requested start/end dates, discount intent, and shipping requirements. If any required information is missing, you will be prompted for only the minimum additional details needed."
        ),
        "few_shot_examples": [
            {
                "Input": {
                    "customerEmailId": "jane.doe@example.com",
                    "receivedDateTime": "2024-06-01T10:15:00Z"
                },
                "Response": {
                    "status": "FAIL",
                    "missingInputs": ["lineItems"],
                    "validationSummary": "customerEmailId and receivedDateTime are valid. Customer found and eligible. Line items missing.",
                    "errors": [
                        {
                            "code": "MISSING_REQUIRED_FIELD",
                            "message": "Line items are required to create a quote. Please provide product/SKU, quantity, and requested dates."
                        }
                    ],
                    "requestId": "REQ-12345",
                    "correlationId": "CORR-67890"
                }
            },
            {
                "Input": {
                    "customerEmailId": "john.smith@example.com",
                    "receivedDateTime": "2024-06-01T09:00:00Z",
                    "lineItems": [
                        {
                            "productSKU": "SKU-001",
                            "quantity": 2,
                            "startDate": "2024-06-10",
                            "endDate": "2024-06-20"
                        }
                    ]
                },
                "Response": {
                    "status": "PASS",
                    "quoteId": "Q-98765",
                    "quoteOrderId": "O-54321",
                    "timestamps": {
                        "quoteCreated": "2024-06-01T09:01:00Z",
                        "orderCreated": "2024-06-01T09:02:00Z"
                    },
                    "warnings": [],
                    "validationSummary": "All required inputs validated. Quote and order created successfully.",
                    "missingInputs": [],
                    "errors": [],
                    "requestId": "REQ-54321",
                    "correlationId": "CORR-09876"
                }
            }
        ]
    }

    # Domain-specific settings
    DOMAIN_SETTINGS = {
        "domain": "general",
        "rag": {
            "enabled": True,
            "retrieval_service": "azure_ai_search",
            "embedding_model": "text-embedding-ada-002",
            "top_k": 5,
            "search_type": "vector_semantic"
        }
    }

    def __init__(self):
        # Load .env if present
        load_dotenv()
        self._validate_env()
        self.crm_api_endpoint = os.getenv("CRM_API_ENDPOINT")
        self.api_auth_token = os.getenv("API_AUTH_TOKEN")
        self.azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.azure_search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.llm_provider = os.getenv("LLM_PROVIDER", self.LLM_DEFAULTS["provider"])
        self.llm_model = os.getenv("LLM_MODEL", self.LLM_DEFAULTS["model"])
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", self.LLM_DEFAULTS["temperature"]))
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", self.LLM_DEFAULTS["max_tokens"]))
        self.system_prompt = self.LLM_DEFAULTS["system_prompt"]
        self.user_prompt_template = self.LLM_DEFAULTS["user_prompt_template"]
        self.few_shot_examples = self.LLM_DEFAULTS["few_shot_examples"]
        self.domain = self.DOMAIN_SETTINGS["domain"]
        self.rag_config = self.DOMAIN_SETTINGS["rag"]

    def _validate_env(self):
        missing = [k for k in self.REQUIRED_ENV_VARS if not os.getenv(k)]
        if missing:
            raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    def get_llm_config(self):
        return {
            "provider": self.llm_provider,
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "few_shot_examples": self.few_shot_examples
        }

    def get_rag_config(self):
        return self.rag_config

    def get_crm_config(self):
        return {
            "crmApiEndpoint": self.crm_api_endpoint,
            "apiAuthToken": self.api_auth_token
        }

    def get_azure_search_config(self):
        return {
            "endpoint": self.azure_search_endpoint,
            "api_key": self.azure_search_api_key,
            "index_name": self.azure_search_index_name
        }

    def get_openai_config(self):
        return {
            "endpoint": self.azure_openai_endpoint,
            "api_key": self.azure_openai_api_key,
            "embedding_deployment": self.azure_openai_embedding_deployment
        }

    def as_dict(self):
        return {
            "llm": self.get_llm_config(),
            "crm": self.get_crm_config(),
            "rag": self.get_rag_config(),
            "azure_search": self.get_azure_search_config(),
            "openai": self.get_openai_config(),
            "domain": self.domain
        }

# Example usage:
# try:
#     config = AgentConfig()
#     print(config.as_dict())
# except ConfigError as e:
#     print(f"Configuration error: {e}")

