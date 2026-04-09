try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 4,
 'runtime_enabled': True,
 'sanitize_pii': True}


import os
import re
import logging
import asyncio
import time as _time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, ValidationError, model_validator
from dotenv import load_dotenv
import requests
from requests.exceptions import RequestException
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# Azure AI Search and OpenAI imports
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

# Observability wrappers (injected automatically by runtime)
# from observability import trace_step, trace_step_sync

# Load environment variables from .env if present
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("CRMQuoteCreationAgent")

# Constants for output contract and fallback
ENHANCED_SYSTEM_PROMPT = (
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
)
FALLBACK_RESPONSE = (
    "The required information could not be found or resolved. Please provide the missing details as indicated, or contact your CRM administrator for assistance."
)

OUTPUT_FORMAT = {
    "status": "PASS | FAIL",
    "quoteId": "<string, present if PASS>",
    "quoteOrderId": "<string, present if PASS>",
    "timestamps": {
        "quoteCreated": "<ISO-8601>",
        "orderCreated": "<ISO-8601>"
    },
    "warnings": ["<warning 1>", "<warning 2>"],
    "validationSummary": "<summary of validations performed>",
    "missingInputs": ["<missing field 1>", "<missing field 2>"],
    "errors": [
        {
            "code": "<ERROR_CODE>",
            "message": "<error message>"
        }
    ],
    "requestId": "<string>",
    "correlationId": "<string>"
}

# -------------------- Configuration Management --------------------

class Config:
    """Configuration loader for environment variables."""
    @staticmethod
    def get(key: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(key, default)

    @staticmethod
    def validate(required_keys: List[str]) -> Tuple[bool, List[str]]:
        missing = [k for k in required_keys if not os.getenv(k)]
        return (len(missing) == 0, missing)

# -------------------- Input Models --------------------

class LineItemModel(BaseModel):
    productSKU: str
    quantity: int
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    discountIntent: Optional[str] = None
    shippingRequirements: Optional[str] = None

    @field_validator("productSKU")
    def validate_productSKU(cls, v):
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("productSKU must be a non-empty string")
        return v.strip()

    @field_validator("quantity")
    def validate_quantity(cls, v):
        if v is None or not isinstance(v, int) or v <= 0:
            raise ValueError("quantity must be a positive integer")
        return v

class QuoteRequestModel(BaseModel):
    customerEmailId: str
    receivedDateTime: str
    lineItems: Optional[List[LineItemModel]] = None
    requestId: Optional[str] = None
    correlationId: Optional[str] = None

    @field_validator("customerEmailId")
    def validate_email(cls, v):
        email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not v or not re.match(email_regex, v):
            raise ValueError("customerEmailId must be a valid email address")
        return v.strip()

    @field_validator("receivedDateTime")
    def validate_datetime(cls, v):
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception:
            raise ValueError("receivedDateTime must be ISO-8601 parseable")
        return v.strip()

    @model_validator(mode="after")
    def check_line_items(cls, values):
        if values.get("lineItems") is not None:
            if not isinstance(values["lineItems"], list) or not values["lineItems"]:
                raise ValueError("lineItems must be a non-empty list if provided")
        return values

# -------------------- Audit Logger --------------------

class AuditLogger:
    """Logs actions, validations, and API calls for auditability."""
    def log_action(self, action: str, details: Dict[str, Any]) -> None:
        masked_details = self._mask_sensitive(details)
        logger.info(f"AuditLog - Action: {action} | Details: {masked_details}")

    def log_error(self, error_code: str, context: Dict[str, Any]) -> None:
        masked_context = self._mask_sensitive(context)
        logger.error(f"AuditLog - Error: {error_code} | Context: {masked_context}")

    def _mask_sensitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        masked = dict(data)
        for key in masked:
            if "email" in key.lower():
                masked[key] = "***@***.***"
            if "token" in key.lower():
                masked[key] = "***MASKED***"
        return masked

# -------------------- Input Validator --------------------

class InputValidator:
    """Validates customerEmailId and receivedDateTime for presence and format."""
    email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"

    def validate_email(self, email: str) -> Tuple[bool, Optional[str]]:
        if not email or not re.match(self.email_regex, email):
            return False, "INVALID_EMAIL"
        return True, None

    def validate_datetime(self, dt: str) -> Tuple[bool, Optional[str]]:
        try:
            datetime.fromisoformat(dt.replace("Z", "+00:00"))
            return True, None
        except Exception:
            return False, "INVALID_DATETIME"

# -------------------- API Integration Layer --------------------

class APIIntegrationLayer:
    """Handles all CRM and business rule API calls, manages authentication and retries."""
    def __init__(self):
        self.crm_api_endpoint = Config.get("CRM_API_ENDPOINT")
        self.api_auth_token = Config.get("API_AUTH_TOKEN")
        self.max_retries = 3
        self.timeout = 30

    def call_crm_api(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.crm_api_endpoint}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_auth_token}",
            "Content-Type": "application/json"
        }
        for attempt in range(self.max_retries):
            try:
                _obs_t0 = _time.time()
                response = requests.post(url, json=data, headers=headers, timeout=self.timeout)
                try:
                    trace_tool_call(
                        tool_name='requests.post',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(response)[:200] if response is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"CRM API call failed: {response.status_code} {response.text}")
            except RequestException as e:
                logger.warning(f"CRM API call exception: {str(e)}")
                _time.sleep(2 ** attempt)
        return {"error": "API_ERROR", "message": "Failed to call CRM API after retries"}

    def call_rules_service(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder for rules service API call
        return {"status": "PASS"}

# -------------------- CRM Data Resolver --------------------

class CRMDataResolver:
    """Resolves customer, account, and required quote/order fields via CRM APIs."""
    def __init__(self, api_layer: APIIntegrationLayer):
        self.api_layer = api_layer

    def resolve_customer(self, email: str) -> Dict[str, Any]:
        endpoint = "/contacts/search"
        data = {"email": email}
        result = self.api_layer.call_crm_api(endpoint, data)
        if "error" in result:
            return {"error": "CUSTOMER_NOT_FOUND", "message": result.get("message", "Customer not found")}
        return {"contactId": result.get("contactId"), "customerStatus": result.get("status", "active")}

    def resolve_account(self, contactId: str) -> Dict[str, Any]:
        endpoint = "/accounts/resolve"
        data = {"contactId": contactId}
        result = self.api_layer.call_crm_api(endpoint, data)
        if "error" in result:
            return {"error": "ACCOUNT_NOT_FOUND", "message": result.get("message", "Account not found")}
        return {
            "accountId": result.get("accountId"),
            "accountStatus": result.get("status", "open"),
            "billTo": result.get("billTo"),
            "shipTo": result.get("shipTo"),
            "currency": result.get("currency"),
            "priceList": result.get("priceList"),
            "paymentTerms": result.get("paymentTerms"),
            "taxRegion": result.get("taxRegion")
        }

# -------------------- Business Rule Engine --------------------

class BusinessRuleEngine:
    """Applies business rules for input validation, customer resolution, quote data resolution, and idempotency."""
    def __init__(self, input_validator: InputValidator, crm_resolver: CRMDataResolver, audit_logger: AuditLogger):
        self.input_validator = input_validator
        self.crm_resolver = crm_resolver
        self.audit_logger = audit_logger

    def apply_rules(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        validation_summary = []
        errors = []
        missing_inputs = []

        # Validate email
        valid_email, email_error = self.input_validator.validate_email(payload.get("customerEmailId", ""))
        if not valid_email:
            errors.append({"code": email_error, "message": "customerEmailId is invalid"})
            missing_inputs.append("customerEmailId")
            validation_summary.append("customerEmailId invalid")
            self.audit_logger.log_error(email_error, payload)
            return {
                "validationSummary": "; ".join(validation_summary),
                "errors": errors,
                "missingInputs": missing_inputs,
                "hard_stop": True
            }

        # Validate datetime
        valid_dt, dt_error = self.input_validator.validate_datetime(payload.get("receivedDateTime", ""))
        if not valid_dt:
            errors.append({"code": dt_error, "message": "receivedDateTime is invalid"})
            missing_inputs.append("receivedDateTime")
            validation_summary.append("receivedDateTime invalid")
            self.audit_logger.log_error(dt_error, payload)
            return {
                "validationSummary": "; ".join(validation_summary),
                "errors": errors,
                "missingInputs": missing_inputs,
                "hard_stop": True
            }

        validation_summary.append("customerEmailId and receivedDateTime are valid.")

        # Resolve customer
        customer_result = self.crm_resolver.resolve_customer(payload["customerEmailId"])
        if "error" in customer_result:
            errors.append({"code": customer_result["error"], "message": customer_result["message"]})
            missing_inputs.append("customerEmailId")
            validation_summary.append("Customer not found.")
            self.audit_logger.log_error(customer_result["error"], payload)
            return {
                "validationSummary": "; ".join(validation_summary),
                "errors": errors,
                "missingInputs": missing_inputs,
                "hard_stop": True
            }
        contactId = customer_result.get("contactId")
        customerStatus = customer_result.get("customerStatus", "active")

        # Resolve account
        account_result = self.crm_resolver.resolve_account(contactId)
        if "error" in account_result:
            errors.append({"code": account_result["error"], "message": account_result["message"]})
            missing_inputs.append("accountId")
            validation_summary.append("Account not found.")
            self.audit_logger.log_error(account_result["error"], payload)
            return {
                "validationSummary": "; ".join(validation_summary),
                "errors": errors,
                "missingInputs": missing_inputs,
                "hard_stop": True
            }
        accountId = account_result.get("accountId")
        accountStatus = account_result.get("accountStatus", "open")

        # Eligibility check (decision table)
        eligibility = "eligible" if customerStatus == "active" and accountStatus == "open" else "not eligible"
        if eligibility != "eligible":
            errors.append({"code": "CUSTOMER_NOT_ELIGIBLE", "message": "Customer/account not eligible for quoting"})
            validation_summary.append("Customer/account not eligible.")
            self.audit_logger.log_error("CUSTOMER_NOT_ELIGIBLE", payload)
            return {
                "validationSummary": "; ".join(validation_summary),
                "errors": errors,
                "missingInputs": [],
                "hard_stop": True
            }

        # Required fields check
        required_fields = [
            "contactId", "accountId", "billTo", "shipTo", "currency", "priceList", "paymentTerms", "taxRegion", "lineItems"
        ]
        resolved_fields = {
            "contactId": contactId,
            "accountId": accountId,
            "billTo": account_result.get("billTo"),
            "shipTo": account_result.get("shipTo"),
            "currency": account_result.get("currency"),
            "priceList": account_result.get("priceList"),
            "paymentTerms": account_result.get("paymentTerms"),
            "taxRegion": account_result.get("taxRegion"),
            "lineItems": payload.get("lineItems")
        }
        for field in required_fields:
            if not resolved_fields.get(field):
                missing_inputs.append(field)
        if missing_inputs:
            errors.append({
                "code": "MISSING_REQUIRED_FIELD",
                "message": f"Missing required fields: {', '.join(missing_inputs)}"
            })
            validation_summary.append("Missing required fields.")
            self.audit_logger.log_error("MISSING_REQUIRED_FIELD", payload)
            return {
                "validationSummary": "; ".join(validation_summary),
                "errors": errors,
                "missingInputs": missing_inputs,
                "hard_stop": True
            }

        validation_summary.append("All required inputs validated. Customer and account eligible.")
        self.audit_logger.log_action("BusinessRuleEngine.apply_rules", resolved_fields)
        return {
            "validationSummary": "; ".join(validation_summary),
            "errors": [],
            "missingInputs": [],
            "resolvedFields": resolved_fields,
            "hard_stop": False
        }

# -------------------- Quote/Order Orchestrator --------------------

class QuoteOrderOrchestrator:
    """Builds quote/order payloads, orchestrates API calls, ensures idempotency."""
    def __init__(self, api_layer: APIIntegrationLayer, business_rule_engine: BusinessRuleEngine):
        self.api_layer = api_layer
        self.business_rule_engine = business_rule_engine
        self.audit_logger = AuditLogger()
        self.idempotency_cache = {}

    def check_idempotency(self, requestId: str) -> Optional[Dict[str, Any]]:
        if not requestId:
            return None
        return self.idempotency_cache.get(requestId)

    def build_quote_payload(self, resolved_data: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "contactId": resolved_data.get("contactId"),
            "accountId": resolved_data.get("accountId"),
            "billTo": resolved_data.get("billTo"),
            "shipTo": resolved_data.get("shipTo"),
            "currency": resolved_data.get("currency"),
            "priceList": resolved_data.get("priceList"),
            "paymentTerms": resolved_data.get("paymentTerms"),
            "taxRegion": resolved_data.get("taxRegion"),
            "lineItems": resolved_data.get("lineItems")
        }
        missing = [k for k, v in payload.items() if not v]
        if missing:
            self.audit_logger.log_error("MISSING_REQUIRED_FIELD", {"missing": missing})
            return {"error": "MISSING_REQUIRED_FIELD", "missingInputs": missing}
        self.audit_logger.log_action("QuoteOrderOrchestrator.build_quote_payload", payload)
        return payload

    def create_quote(self, quotePayload: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = "/quotes/create"
        result = self.api_layer.call_crm_api(endpoint, quotePayload)
        if "error" in result:
            self.audit_logger.log_error("API_ERROR", result)
            return {"error": "API_ERROR", "message": result.get("message", "Failed to create quote")}
        quoteId = result.get("quoteId")
        quoteCreatedTimestamp = result.get("quoteCreatedTimestamp", datetime.utcnow().isoformat() + "Z")
        self.audit_logger.log_action("QuoteOrderOrchestrator.create_quote", {"quoteId": quoteId})
        return {"quoteId": quoteId, "status": "PASS", "quoteCreatedTimestamp": quoteCreatedTimestamp}

    def validate_pricing_availability(self, quotePayload: Dict[str, Any], lineItems: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = self.api_layer.call_rules_service({"quotePayload": quotePayload, "lineItems": lineItems})
        if result.get("status") != "PASS":
            self.audit_logger.log_error("RULES_VALIDATION_FAIL", result)
            return {"error": "RULES_VALIDATION_FAIL", "message": "Pricing/availability/discount validation failed"}
        self.audit_logger.log_action("QuoteOrderOrchestrator.validate_pricing_availability", result)
        return result

    def create_quote_order(self, quoteId: str, lineItems: List[Dict[str, Any]]) -> Dict[str, Any]:
        endpoint = "/quoteorders/create"
        data = {"quoteId": quoteId, "lineItems": lineItems}
        result = self.api_layer.call_crm_api(endpoint, data)
        if "error" in result:
            self.audit_logger.log_error("API_ERROR", result)
            return {"error": "API_ERROR", "message": result.get("message", "Failed to create quote order")}
        quoteOrderId = result.get("quoteOrderId")
        orderCreatedTimestamp = result.get("orderCreatedTimestamp", datetime.utcnow().isoformat() + "Z")
        self.audit_logger.log_action("QuoteOrderOrchestrator.create_quote_order", {"quoteOrderId": quoteOrderId})
        return {"quoteOrderId": quoteOrderId, "status": "PASS", "orderCreatedTimestamp": orderCreatedTimestamp}

# -------------------- Error Handler --------------------

class ErrorHandler:
    """Handles errors, applies retry logic, maps errors to user-friendly messages."""
    def __init__(self, api_layer: APIIntegrationLayer, audit_logger: AuditLogger):
        self.api_layer = api_layer
        self.audit_logger = audit_logger

    def handle_error(self, error_code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.audit_logger.log_error(error_code, context)
        error_map = {
            "INVALID_EMAIL": "The provided email address is invalid.",
            "INVALID_DATETIME": "The provided date/time is invalid.",
            "CUSTOMER_NOT_FOUND": "Customer not found in CRM.",
            "CUSTOMER_NOT_ELIGIBLE": "Customer/account not eligible for quoting.",
            "MISSING_REQUIRED_FIELD": "Required fields are missing.",
            "API_ERROR": "A CRM API error occurred.",
            "DUPLICATE_REQUEST": "Duplicate request detected."
        }
        message = error_map.get(error_code, "An unknown error occurred.")
        return {
            "success": False,
            "error": {
                "code": error_code,
                "message": message
            },
            "tips": "Please check your input and try again. If the issue persists, contact support."
        }

# -------------------- Output Formatter --------------------

class OutputFormatter:
    """Formats responses according to output contract, masks sensitive data."""
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler

    def format_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # Mask PII
        @with_content_safety(config=GUARDRAILS_CONFIG)
        def mask_pii(val):
            if isinstance(val, str) and "@" in val:
                return "***@***.***"
            return val

        response = {}
        response["status"] = result.get("status", "FAIL")
        response["quoteId"] = mask_pii(result.get("quoteId", ""))
        response["quoteOrderId"] = mask_pii(result.get("quoteOrderId", ""))
        response["timestamps"] = result.get("timestamps", {})
        response["warnings"] = result.get("warnings", [])
        response["validationSummary"] = result.get("validationSummary", "")
        response["missingInputs"] = result.get("missingInputs", [])
        response["errors"] = result.get("errors", [])
        response["requestId"] = mask_pii(result.get("requestId", ""))
        response["correlationId"] = mask_pii(result.get("correlationId", ""))
        return response

    def fallback_response(self) -> Dict[str, Any]:
        return {
            "status": "FAIL",
            "validationSummary": "",
            "missingInputs": [],
            "errors": [
                {
                    "code": "INFO_NOT_FOUND",
                    "message": FALLBACK_RESPONSE
                }
            ],
            "requestId": "",
            "correlationId": "",
            "success": False
        }

# -------------------- RAG Retriever --------------------

class RAGRetriever:
    """Retrieves document context from Azure AI Search for knowledge queries."""
    def __init__(self):
        self.search_endpoint = Config.get("AZURE_SEARCH_ENDPOINT")
        self.search_api_key = Config.get("AZURE_SEARCH_API_KEY")
        self.search_index_name = Config.get("AZURE_SEARCH_INDEX_NAME")
        self.openai_endpoint = Config.get("AZURE_OPENAI_ENDPOINT")
        self.openai_api_key = Config.get("AZURE_OPENAI_API_KEY")
        self.embedding_deployment = Config.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        self.top_k = int(Config.get("RAG_TOP_K", "5"))

    def _get_search_client(self) -> Optional[SearchClient]:
        if not all([self.search_endpoint, self.search_api_key, self.search_index_name]):
            logger.warning("Azure Search credentials missing.")
            return None
        return SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.search_index_name,
            credential=AzureKeyCredential(self.search_api_key)
        )

    def _get_openai_client(self) -> Optional[openai.AzureOpenAI]:
        if not all([self.openai_api_key, self.openai_endpoint]):
            logger.warning("Azure OpenAI credentials missing.")
            return None
        return openai.AzureOpenAI(
            api_key=self.openai_api_key,
            api_version="2024-02-01",
            azure_endpoint=self.openai_endpoint
        )

    @trace_agent(agent_name='CRM Quote Creation Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_context(self, query: str) -> List[str]:
        async with trace_step(
            "retrieve_context", step_type="tool_call",
            decision_summary="Retrieve document context from Azure AI Search",
            output_fn=lambda r: f"chunks={len(r) if r else 0}"
        ) as step:
            search_client = self._get_search_client()
            openai_client = self._get_openai_client()
            if not search_client or not openai_client:
                step.capture([])
                return []

            # Embed user query
            try:
                embedding_resp = openai_client.embeddings.create(
                    input=query,
                    model=self.embedding_deployment
                )
                embedding = embedding_resp.data[0].embedding
            except Exception as e:
                logger.warning(f"Embedding creation failed: {str(e)}")
                step.capture([])
                return []

            vector_query = VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=self.top_k,
                fields="vector"
            )
            try:
                results = search_client.search(
                    search_text=query,
                    vector_queries=[vector_query],
                    top=self.top_k,
                    select=["chunk", "title"]
                )
                context_chunks = [r["chunk"] for r in results if r.get("chunk")]
                step.capture(context_chunks)
                return context_chunks
            except Exception as e:
                logger.warning(f"Azure Search retrieval failed: {str(e)}")
                step.capture([])
                return []

# -------------------- LLM Client --------------------

class LLMClient:
    """LLM integration with Azure OpenAI."""
    def __init__(self):
        self.api_key = Config.get("AZURE_OPENAI_API_KEY")
        self.endpoint = Config.get("AZURE_OPENAI_ENDPOINT")
        self.model = Config.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-aba")
        self.temperature = float(Config.get("LLM_TEMPERATURE", "0.7"))
        self.max_tokens = int(Config.get("LLM_MAX_TOKENS", "2000"))

    def get_client(self) -> openai.AsyncOpenAI:
        if not self.api_key or not self.endpoint:
            raise ValueError("Azure OpenAI credentials missing.")
        return openai.AsyncOpenAI(
            api_key=self.api_key,
            api_version="2024-02-01",
            azure_endpoint=self.endpoint
        )

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def chat_completion(self, system_prompt: str, user_message: str, context_chunks: Optional[List[str]] = None) -> str:
        async with trace_step(
            "generate_response", step_type="llm_call",
            decision_summary="Call LLM to produce a reply",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            client = self.get_client()
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            if context_chunks:
                context = "\n".join(context_chunks)
                messages.append({"role": "system", "content": f"Knowledge base context:\n{context}"})
            messages.append({"role": "user", "content": user_message})
            _t0 = _time.time()
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                content = response.choices[0].message.content
                try:
                    trace_model_call(
                        provider="openai",
                        model_name=self.model,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else ""
                    )
                except Exception:
                    pass
                step.capture(content)
                return content
            except Exception as e:
                logger.warning(f"LLM completion failed: {str(e)}")
                step.capture("")
                return ""

# -------------------- Main Agent Class --------------------

class CRMQuoteCreationAgent:
    """Main agent orchestrating quote creation workflow."""
    def __init__(self):
        self.audit_logger = AuditLogger()
        self.input_validator = InputValidator()
        self.api_layer = APIIntegrationLayer()
        self.crm_resolver = CRMDataResolver(self.api_layer)
        self.business_rule_engine = BusinessRuleEngine(self.input_validator, self.crm_resolver, self.audit_logger)
        self.quote_orchestrator = QuoteOrderOrchestrator(self.api_layer, self.business_rule_engine)
        self.error_handler = ErrorHandler(self.api_layer, self.audit_logger)
        self.output_formatter = OutputFormatter(self.error_handler)
        self.rag_retriever = RAGRetriever()
        self.llm_client = LLMClient()

    @trace_agent(agent_name='CRM Quote Creation Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_quote_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with trace_step(
            "process_quote_request", step_type="final",
            decision_summary="End-to-end quote creation workflow",
            output_fn=lambda r: f"status={r.get('status','?')}"
        ) as step:
            # Input validation and business rules
            rule_result = self.business_rule_engine.apply_rules(payload)
            if rule_result.get("hard_stop"):
                formatted = self.output_formatter.format_response(rule_result)
                step.capture(formatted)
                return formatted

            resolved_fields = rule_result.get("resolvedFields", {})
            requestId = payload.get("requestId") or f"REQ-{int(_time.time())}"
            correlationId = payload.get("correlationId") or f"CORR-{int(_time.time())}"

            # Idempotency check
            idempotency_result = self.quote_orchestrator.check_idempotency(requestId)
            if idempotency_result:
                formatted = self.output_formatter.format_response({
                    "status": "PASS",
                    "quoteId": idempotency_result.get("quoteId"),
                    "quoteOrderId": idempotency_result.get("quoteOrderId"),
                    "timestamps": idempotency_result.get("timestamps", {}),
                    "warnings": [],
                    "validationSummary": "Duplicate request detected. Returning existing identifiers.",
                    "missingInputs": [],
                    "errors": [],
                    "requestId": requestId,
                    "correlationId": correlationId
                })
                step.capture(formatted)
                return formatted

            # Build quote payload
            quote_payload = self.quote_orchestrator.build_quote_payload(resolved_fields)
            if "error" in quote_payload:
                formatted = self.output_formatter.format_response({
                    "status": "FAIL",
                    "validationSummary": "Missing required fields.",
                    "missingInputs": quote_payload.get("missingInputs", []),
                    "errors": [{"code": "MISSING_REQUIRED_FIELD", "message": "Missing required fields."}],
                    "requestId": requestId,
                    "correlationId": correlationId
                })
                step.capture(formatted)
                return formatted

            # Create quote
            quote_result = self.quote_orchestrator.create_quote(quote_payload)
            if "error" in quote_result:
                formatted = self.output_formatter.format_response({
                    "status": "FAIL",
                    "validationSummary": "Quote creation failed.",
                    "missingInputs": [],
                    "errors": [{"code": "API_ERROR", "message": quote_result.get("message", "")}],
                    "requestId": requestId,
                    "correlationId": correlationId
                })
                step.capture(formatted)
                return formatted
            quoteId = quote_result.get("quoteId")
            quoteCreatedTimestamp = quote_result.get("quoteCreatedTimestamp")

            # Validate pricing/availability
            pricing_result = self.quote_orchestrator.validate_pricing_availability(quote_payload, resolved_fields.get("lineItems", []))
            if "error" in pricing_result:
                formatted = self.output_formatter.format_response({
                    "status": "FAIL",
                    "validationSummary": "Pricing/availability validation failed.",
                    "missingInputs": [],
                    "errors": [{"code": "RULES_VALIDATION_FAIL", "message": pricing_result.get("message", "")}],
                    "requestId": requestId,
                    "correlationId": correlationId
                })
                step.capture(formatted)
                return formatted

            # Create quote order
            quote_order_result = self.quote_orchestrator.create_quote_order(quoteId, resolved_fields.get("lineItems", []))
            if "error" in quote_order_result:
                formatted = self.output_formatter.format_response({
                    "status": "FAIL",
                    "validationSummary": "Quote order creation failed.",
                    "missingInputs": [],
                    "errors": [{"code": "API_ERROR", "message": quote_order_result.get("message", "")}],
                    "requestId": requestId,
                    "correlationId": correlationId
                })
                step.capture(formatted)
                return formatted
            quoteOrderId = quote_order_result.get("quoteOrderId")
            orderCreatedTimestamp = quote_order_result.get("orderCreatedTimestamp")

            # Cache idempotency
            self.quote_orchestrator.idempotency_cache[requestId] = {
                "quoteId": quoteId,
                "quoteOrderId": quoteOrderId,
                "timestamps": {
                    "quoteCreated": quoteCreatedTimestamp,
                    "orderCreated": orderCreatedTimestamp
                }
            }

            formatted = self.output_formatter.format_response({
                "status": "PASS",
                "quoteId": quoteId,
                "quoteOrderId": quoteOrderId,
                "timestamps": {
                    "quoteCreated": quoteCreatedTimestamp,
                    "orderCreated": orderCreatedTimestamp
                },
                "warnings": [],
                "validationSummary": rule_result.get("validationSummary", ""),
                "missingInputs": [],
                "errors": [],
                "requestId": requestId,
                "correlationId": correlationId
            })
            step.capture(formatted)
            return formatted

    @trace_agent(agent_name='CRM Quote Creation Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def answer_knowledge_query(self, query: str) -> Dict[str, Any]:
        async with trace_step(
            "answer_knowledge_query", step_type="final",
            decision_summary="Answer knowledge query using RAG pipeline",
            output_fn=lambda r: f"length={len(r.get('answer','')) if r else 0}"
        ) as step:
            context_chunks = await self.rag_retriever.retrieve_context(query)
            content = await self.llm_client.chat_completion(
                system_prompt=ENHANCED_SYSTEM_PROMPT,
                user_message=query,
                context_chunks=context_chunks
            )
            if not content:
                answer = FALLBACK_RESPONSE
            else:
                answer = content
            result = {
                "success": True if content else False,
                "answer": answer
            }
            step.capture(result)
            return result

# -------------------- FastAPI App --------------------

app = FastAPI()

agent = CRMQuoteCreationAgent()

@app.post("/quote/create")
@with_content_safety(config=GUARDRAILS_CONFIG)
async def create_quote(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        logger.warning(f"Malformed JSON: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "type": "MalformedJSON",
                    "description": "Malformed JSON request. Please check your syntax (quotes, commas, brackets).",
                    "tips": "Ensure your JSON is valid and does not exceed 50,000 characters."
                }
            }
        )
    try:
        quote_request = QuoteRequestModel(**body)
    except ValidationError as ve:
        logger.warning(f"Input validation error: {ve.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "type": "InputValidationError",
                    "description": "Input validation failed.",
                    "details": ve.errors(),
                    "tips": "Check email format, date/time format, and required fields."
                }
            }
        )
    result = await agent.process_quote_request(quote_request.dict())
    return JSONResponse(
        status_code=200,
        content={"success": True, "result": result}
    )

@app.post("/knowledge/query")
@with_content_safety(config=GUARDRAILS_CONFIG)
async def knowledge_query(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        logger.warning(f"Malformed JSON: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "type": "MalformedJSON",
                    "description": "Malformed JSON request. Please check your syntax (quotes, commas, brackets).",
                    "tips": "Ensure your JSON is valid and does not exceed 50,000 characters."
                }
            }
        )
    query = body.get("query", "")
    if not query or not isinstance(query, str) or not query.strip():
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "type": "InputValidationError",
                    "description": "Query text is required.",
                    "tips": "Provide a non-empty query string."
                }
            }
        )
    result = await agent.answer_knowledge_query(query.strip())
    return JSONResponse(
        status_code=200,
        content=result
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "type": "HTTPException",
                "description": exc.detail
            }
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "type": "InternalServerError",
                "description": "An unexpected error occurred.",
                "tips": "Contact support if the issue persists."
            }
        }
    )

# -------------------- Main Execution --------------------



async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        logger.info("Starting CRM Quote Creation Agent...")
        uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=True)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())