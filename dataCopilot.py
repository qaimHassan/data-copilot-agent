# System imports
import re
import json
import sqlite3
from datetime import datetime
from tqdm.auto import tqdm 
import os
import sys 
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Set, Tuple, Optional, Union, Callable, TypeVar, ParamSpec

# SQL parsing
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Comparison
from sqlparse.tokens import Keyword, DML

# LangChain imports
from langchain_community.llms import Bedrock  # Updated Bedrock import
from langchain_community.chat_models import BedrockChat  # Updated import for Claude 3
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.schema.runnable import RunnablePassthrough

# Data processing
import networkx as nx
import spacy
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np
from collections import defaultdict

# Environment and configuration
from dotenv import load_dotenv

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set up logger
logger = logging.getLogger(__name__)
# loading environment variables
load_dotenv()

P = ParamSpec("P")
T = TypeVar("T")

def tool(
	name: str,
	description: str
) -> Callable[[Callable[P, T]], Callable[P, T]]:
	"""Decorator to define tools for Claude function calling"""
	def decorator(func: Callable[P, T]) -> Callable[P, T]:
		func.tool_name = name
		func.tool_description = description
		return func
	return decorator


class ConceptExtractor:
	def __init__(self, nlp=None):
		self.logger = logging.getLogger(__name__)
		self.nlp = nlp  # Will be passed from ContextManager
		self.vectorizer = TfidfVectorizer(
			max_features=1000,
			stop_words='english',
			ngram_range=(1, 2)
		)
		
	@tool(
		name="extract_concepts_from_context",
		description="Extracts concepts from column context using NLP clustering"
	)
	def extract_concepts_from_context(self, contexts: List[str]) -> Dict[str, List[str]]:
		"""
		Extract concepts from column contexts using clustering
		
		Args:
			contexts: List of context strings from columns
			
		Returns:
			Dictionary mapping concept groups to related terms
		"""
		try:
			# Vectorize contexts
			tfidf_matrix = self.vectorizer.fit_transform(contexts)
			
			# Cluster similar contexts
			clustering = DBSCAN(eps=0.3, min_samples=2).fit(tfidf_matrix.toarray())
			
			# Extract key terms for each cluster
			feature_names = self.vectorizer.get_feature_names_out()
			clusters = defaultdict(list)
			
			for idx, label in enumerate(clustering.labels_):
				if label >= 0:  # Skip noise points (-1)
					# Get top terms for this context
					context_vector = tfidf_matrix[idx].toarray()[0]
					top_term_indices = np.argsort(context_vector)[-5:]  # Top 5 terms
					
					for term_idx in top_term_indices:
						if context_vector[term_idx] > 0:
							clusters[f"concept_cluster_{label}"].append(
								feature_names[term_idx]
							)
			
			return dict(clusters)
			
		except Exception as e:
			logger.error(f"Error extracting concepts from context: {str(e)}")
			return {}

	@tool(
		name="extract_entities_from_text",
		description="Extracts named entities and noun phrases from text"
	)
	def extract_entities_from_text(self, text: str) -> Set[str]:
		"""
		Extract named entities and noun phrases from text
		
		Args:
			text: Input text to analyze
			
		Returns:
			Set of extracted entities and phrases
		"""
		try:
			doc = self.nlp(text)
			concepts = set()
			
			# Extract named entities
			for ent in doc.ents:
				concepts.add(f"entity_{ent.label_.lower()}_{ent.text.lower()}")
			
			# Extract noun phrases
			for chunk in doc.noun_chunks:
				concepts.add(f"phrase_{chunk.text.lower()}")
			
			return concepts
			
		except Exception as e:
			logger.error(f"Error extracting entities: {str(e)}")
			return set()

class SQLMaster:
	@tool(
		name="generate_sql_query",
		description="Generates SQL queries based on natural language input"
	)
	def generate_sql_query(self, query_text: str, context: Dict[str, Any]) -> str:
		# Implementation
		pass

	@tool(
		name="validate_sql",
		description="Validates SQL query structure and syntax"
	)
	def validate_sql(self, query: str) -> Dict[str, Any]:
		# Implementation
		pass

class ETLMaster:
	@tool(
		name="execute_transformation",
		description="Executes data transformation operations"
	)
	def execute_transformation(self, operation: str, data: Any) -> Any:
		# Implementation
		pass

class GoalContextSearcher:
	@tool(
		name="search_relevant_context",
		description="Searches for context relevant to a specified goal"
	)
	def search_relevant_context(self, goal: str) -> Dict[str, Any]:
		# Implementation
		pass

# older Classes

class IntentType(Enum):
	GENERAL = "general"         # General conversation, greetings
	SQL = "sql"                 # SQL query writing/modification
	DISCOVERY = "discovery"     # Data discovery, finding attributes
	GOAL = "goal"              # Goal-based exploration
	EXECUTION = "execution"     # Query execution, view creation

@dataclass
class Intent:
	type: IntentType
	confidence: float
	context: Dict[str, Any]
	requires_context: bool
	subtype: Optional[str] = None  # For more specific categorization

class IntentProcessor:
	def __init__(self, llm_interface, context_manager):
		"""Initialize IntentProcessor with required components
		
		Args:
			llm_interface: LLMInterface instance for generating responses
			context_manager: ContextManager instance for handling data context
		"""
		self.llm = llm_interface
		self.context_manager = context_manager
		self.logger = logging.getLogger(__name__)
		self.conversation_history = []

	def process_general_intent(self, user_input: str, intent: Intent) -> str:
		"""Handles general conversation"""
		try:
			system_prompt = """Your name is "J" (From Complex Numbers iota & j), a friendly and helpful data analysis assistant. 
			Respond naturally and warmly to general queries while maintaining professionalism.
			For greetings, always introduce yourself as J.
			Never mention Claude or Anthropic."""
			
			# Generate response using LLM
			response = self.llm.generate_response(
				prompt_data={
					"input": user_input,
					"system_prompt": system_prompt
				},
				model_type='general'
			)
			
			# Extract the actual response content
			if isinstance(response, LLMResponse):
				return response.content
			return str(response)
			
		except Exception as e:
			self.logger.error(f"Error processing general intent: {str(e)}")
			return "I apologize, but I encountered an error. How else can I help you?"

	def detect_intent(self, user_input: str) -> Intent:
		"""Detects the intent of the user input"""
		try:
			# Create intent detection prompt
			prompt = {
				"input": f"Analyze this user query and determine the intent: {user_input}\n\n" +
						"Respond in this exact format:\n" +
						"{\n" +
						'  "intent": "GENERAL/SQL/DISCOVERY/GOAL/EXECUTION",\n' +
						'  "confidence": 0.0-1.0,\n' +
						'  "requires_context": true/false,\n' +
						'  "reasoning": "brief explanation"\n' +
						"}"
			}
			
			# Generate intent detection response
			response = self.llm.generate_response(
				prompt_data=prompt,
				model_type='analysis'
			)
			
			# Parse JSON response with error handling
			try:
				if isinstance(response, LLMResponse):
					parsed_response = json.loads(response.content)
				else:
					parsed_response = json.loads(response)
			except json.JSONDecodeError:
				# Fallback to GENERAL intent if parsing fails
				return Intent(
					type=IntentType.GENERAL,
					confidence=1.0,
					context={"reasoning": "Failed to parse intent detection response"},
					requires_context=False
				)
			
			# Create and return Intent object
			return Intent(
				type=IntentType[parsed_response["intent"]],
				confidence=float(parsed_response["confidence"]),
				context={"reasoning": parsed_response["reasoning"]},
				requires_context=parsed_response["requires_context"],
				subtype=None
			)
			
		except Exception as e:
			self.logger.error(f"Error in intent detection: {str(e)}")
			# Default to general intent if detection fails
			return Intent(
				type=IntentType.GENERAL,
				confidence=1.0,
				context={"error": str(e)},
				requires_context=False
			)

	def process_general_intent(self, user_input: str, intent: Intent) -> str:
		"""Handles general conversation"""
		try:
			system_prompt = """You are a friendly and helpful data analysis assistant. 
			Respond naturally and warmly to general queries while maintaining professionalism.
			For greetings, respond in a friendly manner."""
			
			response = self.llm.generate_response({
				"input": user_input,
				"system_prompt": system_prompt
			})
			
			# Make sure we return the actual response content
			return response.content if isinstance(response, LLMResponse) else str(response)
			
		except Exception as e:
			self.logger.error(f"Error processing general intent: {str(e)}")
			return f"I apologize, but I encountered an error: {str(e)}"

		
	def _create_intent_detection_prompt(self) -> ChatPromptTemplate:
		"""Creates the prompt template for intent detection"""
		return ChatPromptTemplate.from_messages([
			SystemMessage(content="""You are an intent detection system for a data analysis copilot. 
			Analyze user queries and classify them into one of these categories:
			
			1. GENERAL: General conversation, greetings, or non-data queries
			   Examples: "Hello", "How are you?", "What's the weather like?"
			
			2. SQL: Requests for SQL query writing or modification
			   Examples: "Write a query to get sales data", "Show me customer orders"
			
			3. DISCOVERY: Data discovery and attribute search
			   Examples: "What columns do we have for customer data?", "Where can I find revenue information?"
			
			4. GOAL: Goal-based data exploration
			   Examples: "I want to build a churn prediction model", "Help me analyze sales trends"
			
			5. EXECUTION: Query execution and data manipulation
			   Examples: "Create a view of active customers", "Run this query", "Show me the data"
			
			Return response in this exact format:
			{
				"intent": "INTENT_TYPE",
				"confidence": CONFIDENCE_SCORE,
				"requires_context": true/false,
				"subtype": "SPECIFIC_SUBTYPE",
				"reasoning": "BRIEF_EXPLANATION"
			}"""),
			HumanMessage(content="{input}")
		])

	def process_intent(self, intent: Intent, user_input: str) -> str:
		"""Routes the intent to appropriate processor"""
		try:
			# Process based on intent type
			if intent.confidence < 0.7:
				return self._handle_low_confidence_intent(intent, user_input)
				
			# Process using appropriate handler
			if intent.type == IntentType.GENERAL:
				return self.process_general_intent(user_input, intent)
			elif intent.type == IntentType.DISCOVERY:
				return self.process_discovery_intent(user_input, intent)
			elif intent.type == IntentType.SQL:
				return self.process_sql_intent(user_input, intent)
			elif intent.type == IntentType.GOAL:
				return self.process_goal_intent(user_input, intent)
			elif intent.type == IntentType.EXECUTION:
				return self.process_execution_intent(user_input, intent)
				
		except Exception as e:
			self.logger.error(f"Error processing intent: {str(e)}")
			return f"I apologize, but I encountered an error while processing your request. {str(e)}"


	def process_sql_intent(self, user_input: str, intent: Intent) -> str:
		"""Handles SQL query generation and modification"""
		if intent.requires_context:
			context = self.context_manager.get_focused_context(user_input)
			related_columns = self.context_manager.get_related_columns(user_input)
			
			system_prompt = f"""Given this database context:
			{context}
			
			And these related columns: {related_columns}

			based on {user_input} search for all relevant columns/attributes across all databases, tables, columns
			and list them as 
			
			db_1:
				table_1 -> column_1
				table_1 -> column_2
				...
				table_2 -> column_1
				...
			db_2:
				table_1 -> column_1
				...
			...
			
			Generate a SQL query that addresses: {user_input} & 100% accurate to the listed relevant data"""
		else:
			system_prompt = "Help with SQL query modification or explanation."
			
		response = self.llm.generate_sql({
			"input": user_input,
			"system_prompt": system_prompt,
			"context": context if intent.requires_context else None
		})
		
		return response


	def process_discovery_intent(self, user_input: str, intent: Intent) -> str:
		"""Handles data discovery requests"""
		try:
			# Get focused context from knowledge graph and semantic search
			focused_context = self.context_manager.get_focused_context(user_input)
			
			# If no context found, try related columns
			if not focused_context:
				focused_context = self.context_manager.get_related_columns(user_input)
			
			# Create discovery prompt
			system_prompt = f"""Based on the available database schema and context:
			{focused_context}
			
			Help the user discover relevant data attributes and answer their question about data availability.
			Provide a clear, structured response showing the relevant attributes organized by database and table in the 
			following format


				db_1:
					table_1 -> column_1
					table_1 -> column_2
					...
					table_2 -> column_1
					...
				db_2:
					table_1 -> column_1
					...
				...

				"""
			
			# Generate response using the context
			response = self.llm.generate_response({
				"input": user_input,
				"system_prompt": system_prompt,
				"context": focused_context
			})
			
			return response.content
		
		except Exception as e:
			self.logger.error(f"Error in discovery intent: {str(e)}")
			return f"I apologize, but I encountered an error while discovering attributes: {str(e)}"

	def process_goal_intent(self, user_input: str, intent: Intent) -> str:
		"""Handles goal-based exploration"""
		# Get comprehensive context for goal-based analysis
		context = self.context_manager.get_focused_context(user_input)
		
		system_prompt = f"""As a data analysis expert, help achieve this goal by discoverying relevant attributes accross all DBs: {user_input}
		
		Available data: {context}
		Goal: extract goal/objective from {user_input}
		
		Provide a structured plan including:
		1. Relevant data attributes from Available data only: search for all relevant columns/attributes across all databases, tables, columns
		and list them as 
			
			db_1:
				table_1 -> column_1
				table_1 -> column_2
				...
				table_2 -> column_1
				...
			db_2:
				table_1 -> column_1
				...
			...

		2. Suggested analysis approach
		3. Potential challenges
		4. Recommend additional attributes or public data if specifically asked -> Show it as RECOMMENDED ADDITIONAL DATA: <provide source>
		5. Next steps
		6. Write python code with input variables and output variables if applicable and only if specifically asked
		"""
		
		response = self.llm.generate_response({
			"input": user_input,
			"system_prompt": system_prompt,
			"context": context
		})
		
		return response.content

	def process_execution_intent(self, user_input: str, intent: Intent) -> str:
		"""Handles query execution requests"""
		# Validate and execute queries
		try:
			# First, validate the query or execution request
			validation_result = self.context_manager.validate_query_context(user_input)
			
			if not validation_result['is_valid']:
				return f"Query validation failed: {validation_result['reason']}"
				
			# Process the execution request
			result = self.query_processor.execute_query(user_input)
			
			return self.format_execution_result(result)
			
		except Exception as e:
			logger.error(f"Query execution error: {str(e)}")
			return f"Error executing query: {str(e)}"

	def _handle_low_confidence_intent(self, intent: Intent, user_input: str) -> str:
		"""Handles cases where intent detection confidence is low"""
		clarification_prompt = f"""I'm not quite sure what you're asking about. 
		Are you trying to:
		1. Write or modify a SQL query?
		2. Find specific data attributes?
		3. Analyze data for a specific goal?
		4. Execute a query or create a view?
		
		Please clarify your request."""
		
		return clarification_prompt

	def format_execution_result(self, result: Any) -> str:
		"""Formats the execution result for display"""
		if isinstance(result, pd.DataFrame):
			return result.to_string()
		elif isinstance(result, dict):
			return json.dumps(result, indent=2)
		else:
			return str(result)

@dataclass
class LLMResponse:
	content: str
	metadata: Dict[str, Any]
	raw_response: Any = None

class PromptLibrary:
	"""Centralized storage for system prompts"""
	
	SQL_GENERATION = """You are an expert SQL query writer. Given the database schema and context, 
	write efficient and accurate SQL queries. Always consider:
	1. Proper table and column references
	2. Appropriate joins when needed
	3. Optimal filtering conditions
	4. Clear formatting with proper indentation
	
	Available Schema:
	{schema}
	
	If the requested query involves tables or columns not in the schema, explain what's missing."""

	CONTEXT_ENHANCEMENT = """You are a data context expert. Analyze the provided information and enhance it with:
	1. Relevant relationships between data elements
	2. Potential use cases
	3. Data quality considerations
	4. Related business context
	
	Current Context:
	{context}"""

	DATA_DISCOVERY = """You are a data discovery assistant. Help users understand and explore their data by:
	1. Identifying relevant data elements
	2. Explaining relationships
	3. Suggesting useful combinations
	4. Highlighting limitations
	
	Available Data:
	{context}"""

class LLMInterface:
	def __init__(self):
		"""Initialize LLM interface with different model configurations"""
		self.logger = logging.getLogger(__name__)
		
		# Initialize Bedrock Claude models
		self.models = {
			'general': BedrockChat(
				model_id="anthropic.claude-3-sonnet-20240229-v1:0",
				model_kwargs={
					"temperature": 0.7,
					"max_tokens": 4096,
					"anthropic_version": "bedrock-2023-05-31"
				},
				region_name="us-east-1",
				credentials_profile_name="default"
			),
			'sql': BedrockChat(
				model_id="anthropic.claude-3-sonnet-20240229-v1:0",
				model_kwargs={
					"temperature": 0.1,
					"max_tokens": 4096,
					"anthropic_version": "bedrock-2023-05-31"
				},
				region_name="us-east-1"
			),
			'analysis': BedrockChat(
				model_id="anthropic.claude-3-sonnet-20240229-v1:0",
				model_kwargs={
					"temperature": 0.3,
					"max_tokens": 4096,
					"anthropic_version": "bedrock-2023-05-31"
				},
				region_name="us-east-1"
			)
		}
		
		# Initialize conversation memory
		self.memory = ConversationBufferMemory(
			return_messages=True,
			memory_key="chat_history"
		)
		
		# Initialize prompt templates
		self.prompt_templates = {
			'general': ChatPromptTemplate.from_messages([
				SystemMessage(content="""You are a friendly and helpful data analysis assistant.
				You help users understand and analyze their data, provide insights, and assist with queries.
				Maintain a professional yet conversational tone."""),
				MessagesPlaceholder(variable_name="chat_history"),
				HumanMessage(content="{query}")
			]),
			'sql': ChatPromptTemplate.from_messages([
				SystemMessage(content="""You are an expert SQL query writer.
				Help users write efficient and accurate SQL queries.
				Available Schema: {schema}"""),
				MessagesPlaceholder(variable_name="chat_history"),
				HumanMessage(content="{query}")
			])
		}

	def generate_response(
		self,
		prompt_data: Dict[str, Any],
		model_type: str = 'general'
	) -> LLMResponse:
		"""Generate response using the specified model type"""
		try:
			# Select appropriate model
			model = self.models.get(model_type, self.models['general'])
			
			# Create messages for the chat
			messages = [
				SystemMessage(content=prompt_data.get("system_prompt", "You are a helpful assistant.")),
				HumanMessage(content=prompt_data.get("input", ""))
			]
			
			# Generate response
			response = model(messages)
			
			# Update conversation memory
			self.memory.save_context(
				{"input": prompt_data.get("input", "")},
				{"output": response.content}
			)
			
			return LLMResponse(
				content=response.content,
				metadata={
					"model_type": model_type,
					"prompt_data": prompt_data
				},
				raw_response=response
			)
			
		except Exception as e:
			self.logger.error(f"Error generating response: {str(e)}")
			raise

	def generate_sql(
		self,
		context: Dict[str, Any],
		query_intent: str
	) -> LLMResponse:
		"""Generate SQL query based on context and intent"""
		try:
			prompt_data = {
				"schema": self._format_schema_context(context),
				"input": query_intent
			}
			
			return self.generate_response(prompt_data, model_type='sql')
			
		except Exception as e:
			self.logger.error(f"Error generating SQL: {str(e)}")
			raise

	def enhance_context(
		self,
		base_context: Dict[str, Any],
		enhancement_type: str = 'general'
	) -> LLMResponse:
		"""
		Enhances provided context with additional insights
		"""
		try:
			prompt_data = {
				"context": base_context,
				"enhancement_type": enhancement_type,
				"input": json.dumps(base_context)
			}
			
			return self.generate_response(prompt_data, model_type='analysis')
			
		except Exception as e:
			self.logger.error(f"Error enhancing context: {str(e)}")
			raise

	def _create_prompt_template(
		self,
		prompt_data: Dict[str, Any],
		model_type: str
	) -> ChatPromptTemplate:
		"""Creates appropriate prompt template based on model type"""
		messages = []
		
		# Add system message based on model type
		if model_type == 'sql':
			messages.append(SystemMessage(
				content=self.prompt_library.SQL_GENERATION.format(
					schema=prompt_data.get("schema", "")
				)
			))
		elif model_type == 'analysis':
			messages.append(SystemMessage(
				content=self.prompt_library.CONTEXT_ENHANCEMENT.format(
					context=prompt_data.get("context", "")
				)
			))
		else:
			messages.append(SystemMessage(
				content=prompt_data.get("system_prompt", "You are a helpful assistant.")
			))
		
		# Add chat history and user input
		messages.extend([
			MessagesPlaceholder(variable_name="chat_history"),
			HumanMessage(content="{input}")
		])
		
		return ChatPromptTemplate.from_messages(messages)

	def _format_schema_context(self, context: Dict[str, Any]) -> str:
		"""Formats database schema context for SQL generation"""
		formatted_context = []
		
		for db_name, db_info in context.items():
			formatted_context.append(f"\nDatabase: {db_name}")
			
			for table_name, table_info in db_info.items():
				formatted_context.append(f"\n  Table: {table_name}")
				
				for col_name, col_info in table_info.items():
					formatted_context.append(
						f"    - {col_name} ({col_info['data_type']}): {col_info['context']}"
					)
		
		return "\n".join(formatted_context)

	def _get_tools_for_model_type(self, model_type: str) -> List[Dict[str, Any]]:
		"""Get appropriate tools based on model type"""
		tools = []
		
		if model_type == 'sql':
			tools.append({
				"type": "function",
				"function": {
					"name": "generate_sql",
					"description": "Generate SQL query from natural language",
					"parameters": {
						"type": "object",
						"properties": {
							"query": {
								"type": "string",
								"description": "The SQL query to generate"
							},
							"tables": {
								"type": "array",
								"items": {"type": "string"},
								"description": "Available tables"
							}
						},
						"required": ["query"]
					}
				}
			})
		elif model_type == 'discovery':
			tools.append({
				"type": "function",
				"function": {
					"name": "discover_data",
					"description": "Find relevant data attributes",
					"parameters": {
						"type": "object",
						"properties": {
							"search_terms": {
								"type": "array",
								"items": {"type": "string"},
								"description": "Terms to search for"
							}
						},
						"required": ["search_terms"]
					}
				}
			})
		
		return tools

	def clear_conversation_history(self):
		"""Clears the conversation history"""
		self.memory.clear()

	def get_conversation_history(self) -> List[Dict[str, str]]:
		"""Returns the current conversation history"""
		return self.memory.load_memory_variables({})["chat_history"]

	def add_context_to_history(self, context: Dict[str, Any]):
		"""Adds context information to conversation history"""
		self.memory.save_context(
			{"input": "Context Update"},
			{"output": json.dumps(context)}
		)


@dataclass
class ColumnInfo:
	database: str
	table: str
	column: str
	data_type: str
	context: str
	metadata: Dict[str, Any] = None

@dataclass
class SearchResult:
	column_info: ColumnInfo
	relevance_score: float
	match_context: str


class ContextManager:
	def __init__(self, context_info: Dict):
		"""Initialize Context Manager with database schema and context information
		
		Args:
			context_info: Dictionary containing database schema and context
		"""
		# Initialize logger first
		self.logger = logging.getLogger(__name__)
		
		# Initialize NLP components first
		self.logger.info("Loading NLP models...")
		self.nlp = spacy.load("en_core_web_sm")
		self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
		
		# Initialize core components
		self.context_info = context_info
		self.concept_extractor = ConceptExtractor()
		self.column_cache = {}  # Cache for frequently accessed columns
		
		# Initialize concept clusters
		self.logger.info("Initializing concept clusters...")
		self.concept_clusters = self._initialize_concept_clusters()
		self.logger.info("Concept clusters initialized")
		
		# Create knowledge graph
		self.logger.info("Creating knowledge graph...")
		self.knowledge_graph = self._create_knowledge_graph()
		self.logger.info("Knowledge graph created")
		
		# Create semantic index last
		self.logger.info("Creating semantic index...")
		self.semantic_index = self._create_semantic_index()
		self.logger.info("Semantic index created")
		
	@tool(
		name="initialize_concept_clusters",
		description="Initialize concept clusters from column contexts"
	)
	def _initialize_concept_clusters(self) -> Dict[str, List[str]]:
		"""Initialize concept clusters from all column contexts"""
		try:
			# Collect all column contexts
			contexts = []
			for db_info in self.context_info.values():
				for table_info in db_info.values():
					for col_info in table_info.values():
						context = col_info.get('context', '')
						if context:
							contexts.append(context)
			
			# Extract concepts using clustering
			return self.concept_extractor.extract_concepts_from_context(contexts)
			
		except Exception as e:
			logger.error(f"Error initializing concept clusters: {str(e)}")
			return {}

	@tool(
		name="search_knowledge_graph",
		description="Search knowledge graph for relevant nodes"
	)
	def _graph_search(self, query: str) -> List[SearchResult]:
		"""Performs knowledge graph-based search"""
		try:
			results = []
			
			# Extract concepts from query
			query_concepts = self.concept_extractor.extract_entities_from_text(query)
			
			# Match query concepts with clustered concepts
			matched_clusters = set()
			for cluster_id, terms in self.concept_clusters.items():
				if any(term in query.lower() for term in terms):
					matched_clusters.add(cluster_id)
			
			# Find relevant nodes in knowledge graph
			for concept in query_concepts | matched_clusters:
				if self.knowledge_graph.has_node(concept):
					for node in nx.single_source_shortest_path_length(
						self.knowledge_graph, concept, cutoff=2
					):
						if self.knowledge_graph.nodes[node]['type'] == 'column':
							parts = node.split('.')
							if len(parts) == 3:
								db, table, column = parts
								node_data = self.knowledge_graph.nodes[node]
								
								results.append(SearchResult(
									column_info=ColumnInfo(
										database=db,
										table=table,
										column=column,
										data_type=node_data['data_type'],
										context=node_data['context']
									),
									relevance_score=1.0,
									match_context=f"concept_match_{concept}"
								))
			
			return results
			
		except Exception as e:
			logger.error(f"Error in graph search: {str(e)}")
			return []

	@tool(
		name="create_knowledge_graph",
		description="Create knowledge graph from context information"
	)
	def _create_knowledge_graph(self) -> nx.Graph:
		"""Creates knowledge graph from context information"""
		G = nx.Graph()
		
		try:
			# Add database and table nodes
			for db_name, db_info in self.context_info.items():
				G.add_node(db_name, type='database')
				
				for table_name, table_info in db_info.items():
					table_node = f"{db_name}.{table_name}"
					G.add_node(table_node, type='table')
					G.add_edge(db_name, table_node)
					
					# Process columns and their contexts
					for col_name, col_info in table_info.items():
						col_node = f"{db_name}.{table_name}.{col_name}"
						
						data_type = col_info.get('dt', col_info.get('data_type', 'unknown'))
						context = col_info.get('c', col_info.get('context', ''))
						
						G.add_node(col_node, 
								type='column',
								data_type=data_type,
								context=context)
						G.add_edge(table_node, col_node)
						
						# Extract and add concept nodes from column context
						if context:
							# Get concepts from text
							text_concepts = self.concept_extractor.extract_entities_from_text(context)
							
							# Find matching concept clusters
							for cluster_id, terms in self.concept_clusters.items():
								if any(term in context.lower() for term in terms):
									concept_node = f"concept_{cluster_id}"
									if not G.has_node(concept_node):
										G.add_node(concept_node, type='concept')
									G.add_edge(col_node, concept_node)
							
							# Add text-based concepts
							for concept in text_concepts:
								if not G.has_node(concept):
									G.add_node(concept, type='concept')
								G.add_edge(col_node, concept)
			
			return G
			
		except Exception as e:
			logger.error(f"Error creating knowledge graph: {str(e)}")
			raise


	def _extract_concepts(self, text: str) -> Set[str]:
		"""
		Extracts key concepts from text using NLP
		"""
		try:
			doc = self.nlp(text)
			concepts = set()
			
			# Extract named entities
			for ent in doc.ents:
				concepts.add(f"concept_{ent.label_.lower()}_{ent.text.lower()}")
			
			# Extract noun phrases
			for chunk in doc.noun_chunks:
				concepts.add(f"concept_noun_{chunk.text.lower()}")
				
			# Extract key terms (dates, numbers, etc.)
			for token in doc:
				if token.like_num:
					concepts.add("concept_numeric")
				# Check for date-like patterns without using like_date
				if token.pos_ == "NUM" and any(date_word in text.lower() 
					for date_word in ["date", "year", "month", "day"]):
					concepts.add("concept_date")
				# Check for currency symbols or terms
				if token.text in ["$", "€", "£"] or token.text.lower() in ["usd", "eur", "gbp"]:
					concepts.add("concept_currency")
				
			# Add domain-specific concepts
			if any(term in text.lower() for term in ['revenue', 'sales', 'profit']):
				concepts.add("concept_financial")
			if any(term in text.lower() for term in ['customer', 'client', 'user']):
				concepts.add("concept_customer")
			if any(term in text.lower() for term in ['employee', 'staff', 'personnel']):
				concepts.add("concept_employee")
				
			return concepts
			
		except Exception as e:
			self.logger.error(f"Error extracting concepts: {str(e)}")
			return set()  # Return empty set on error
			
	def _create_knowledge_graph(self) -> nx.Graph:
		"""Creates a knowledge graph from the context information"""
		G = nx.Graph()
		
		try:
			# Add database nodes
			for db_name, db_info in self.context_info.items():
				G.add_node(db_name, type='database')
				
				# Add table nodes
				for table_name, table_info in db_info.items():
					table_node = f"{db_name}.{table_name}"
					G.add_node(table_node, type='table')
					G.add_edge(db_name, table_node)
					
					# Add column nodes
					for col_name, col_info in table_info.items():
						col_node = f"{db_name}.{table_name}.{col_name}"
						
						# Get data type, handling both 'dt' and 'data_type' keys
						data_type = col_info.get('dt', col_info.get('data_type', 'unknown'))
						
						# Get context, handling both 'c' and 'context' keys
						context = col_info.get('c', col_info.get('context', ''))
						
						G.add_node(col_node, 
								 type='column',
								 data_type=data_type,
								 context=context)
						G.add_edge(table_node, col_node)
						
						# Extract and add concept nodes
						concepts = self._extract_concepts(context)
						for concept in concepts:
							if not G.has_node(concept):
								G.add_node(concept, type='concept')
							G.add_edge(col_node, concept)
			
			self.logger.info("Knowledge graph created successfully")
			return G
			
		except Exception as e:
			self.logger.error(f"Error creating knowledge graph: {str(e)}")
			raise

	def _create_semantic_index(self) -> Dict[str, Any]:
		"""Creates semantic index for efficient context retrieval"""
		try:
			context_items = []
			context_texts = []
			
			for db_name, db_info in self.context_info.items():
				for table_name, table_info in db_info.items():
					for col_name, col_info in table_info.items():
						# Get context, handling both formats
						context_text = col_info.get('c', col_info.get('context', ''))
						# Get data type, handling both formats
						data_type = col_info.get('dt', col_info.get('data_type', 'unknown'))
						
						context_items.append(ColumnInfo(
							database=db_name,
							table=table_name,
							column=col_name,
							data_type=data_type,
							context=context_text
						))
						context_texts.append(context_text)
			
			# Generate embeddings
			embeddings = self.embedding_model.encode(context_texts, 
												   show_progress_bar=False,
												   batch_size=32)
			
			return {
				'items': context_items,
				'embeddings': embeddings
			}
			
		except Exception as e:
			self.logger.error(f"Error creating semantic index: {str(e)}")
			raise

	def get_focused_context(self, query: str) -> Dict[str, Any]:
		"""
		Gets focused context relevant to the query
		
		Args:
			query: User's query string
		
		Returns:
			Dictionary containing relevant context information
		"""
		try:
			# Get semantic search results
			semantic_results = self._semantic_search(query)
			
			# Get knowledge graph results
			graph_results = self._graph_search(query)
			
			# Combine and deduplicate results
			combined_results = self._combine_search_results(
				semantic_results, 
				graph_results
			)
			
			# Format results hierarchically
			formatted_context = self._format_hierarchical_context(combined_results)
			
			return formatted_context
			
		except Exception as e:
			self.logger.error(f"Error getting focused context: {str(e)}")
			return {"error": str(e)}

	def _semantic_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
		"""Performs semantic search on context"""
		try:
			# Generate query embedding
			query_embedding = self.embedding_model.encode([query])[0]
			
			# Calculate similarities
			similarities = cosine_similarity(
				[query_embedding], 
				self.semantic_index['embeddings']
			)[0]
			
			# Get top results
			top_indices = np.argsort(similarities)[-top_k:][::-1]
			
			results = []
			for idx in top_indices:
				if similarities[idx] > 0.3:  # Relevance threshold
					results.append(SearchResult(
						column_info=self.semantic_index['items'][idx],
						relevance_score=float(similarities[idx]),
						match_context="semantic_similarity"
					))
			
			return results
			
		except Exception as e:
			self.logger.error(f"Error in semantic search: {str(e)}")
			return []

	def _graph_search(self, query: str) -> List[SearchResult]:
		"""Performs knowledge graph-based search"""
		try:
			results = []
			query_concepts = self._extract_concepts(query)
			
			# Find columns connected to matching concepts
			for concept in query_concepts:
				if self.knowledge_graph.has_node(concept):
					for node in nx.single_source_shortest_path_length(
						self.knowledge_graph, concept, cutoff=2
					):
						if self.knowledge_graph.nodes[node]['type'] == 'column':
							db, table, column = node.split('.')
							node_data = self.knowledge_graph.nodes[node]
							
							results.append(SearchResult(
								column_info=ColumnInfo(
									database=db,
									table=table,
									column=column,
									data_type=node_data['data_type'],
									context=node_data['context']
								),
								relevance_score=1.0,
								match_context=f"concept_match_{concept}"
							))
			
			return results
			
		except Exception as e:
			self.logger.error(f"Error in graph search: {str(e)}")
			return []

	def _combine_search_results(
		self, 
		semantic_results: List[SearchResult],
		graph_results: List[SearchResult]
	) -> List[SearchResult]:
		"""Combines and deduplicates search results"""
		combined = {}
		
		# Add semantic results
		for result in semantic_results:
			key = (result.column_info.database, 
				  result.column_info.table, 
				  result.column_info.column)
			if key not in combined or result.relevance_score > combined[key].relevance_score:
				combined[key] = result
		
		# Add graph results
		for result in graph_results:
			key = (result.column_info.database, 
				  result.column_info.table, 
				  result.column_info.column)
			if key not in combined or result.relevance_score > combined[key].relevance_score:
				combined[key] = result
		
		return list(combined.values())

	def get_related_columns(self, column_name: str) -> List[ColumnInfo]:
		"""Gets related columns based on knowledge graph connections"""
		try:
			if column_name in self.column_cache:
				return self.column_cache[column_name]
			
			related_columns = []
			
			# Find columns connected through common concepts
			if self.knowledge_graph.has_node(column_name):
				# Get connected concepts
				concepts = [n for n in self.knowledge_graph.neighbors(column_name)
						  if self.knowledge_graph.nodes[n]['type'] == 'concept']
				
				# Find columns connected to these concepts
				for concept in concepts:
					for node in self.knowledge_graph.neighbors(concept):
						if (self.knowledge_graph.nodes[node]['type'] == 'column' 
							and node != column_name):
							db, table, column = node.split('.')
							node_data = self.knowledge_graph.nodes[node]
							
							related_columns.append(ColumnInfo(
								database=db,
								table=table,
								column=column,
								data_type=node_data['data_type'],
								context=node_data['context']
							))
			
			# Cache the results
			self.column_cache[column_name] = related_columns
			return related_columns
			
		except Exception as e:
			self.logger.error(f"Error getting related columns: {str(e)}")
			return []

	def validate_query_context(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""
		Validates if the query uses correct database context
		
		Args:
			query: SQL query string
			context: Optional context dictionary
			
		Returns:
			Dictionary with validation results
		"""
		try:
			# Extract table and column references from query
			referenced_elements = self._extract_query_elements(query)
			
			# Validate against knowledge graph
			invalid_elements = []
			for element in referenced_elements:
				if not self._validate_element(element):
					invalid_elements.append(element)
			
			return {
				'is_valid': len(invalid_elements) == 0,
				'invalid_elements': invalid_elements,
				'reason': f"Invalid elements found: {invalid_elements}" if invalid_elements else None
			}
			
		except Exception as e:
			self.logger.error(f"Error in query validation: {str(e)}")
			return {
				'is_valid': False,
				'reason': f"Validation error: {str(e)}"
			}

	def _extract_query_elements(self, query: str) -> List[str]:
		"""Extracts database elements (tables, columns) from query"""
		# This is a simplified version - in practice, you'd want to use a proper SQL parser
		elements = []
		# Implementation details...
		return elements

	def _validate_element(self, element: str) -> bool:
		"""Validates if an element exists in the knowledge graph"""
		return self.knowledge_graph.has_node(element)

	def _format_hierarchical_context(self, results: List[SearchResult]) -> Dict[str, Any]:
		"""Formats search results into a hierarchical structure"""
		hierarchy = {}
		
		for result in results:
			db = result.column_info.database
			table = result.column_info.table
			column = result.column_info.column
			
			if db not in hierarchy:
				hierarchy[db] = {}
			if table not in hierarchy[db]:
				hierarchy[db][table] = {}
				
			hierarchy[db][table][column] = {
				'data_type': result.column_info.data_type,
				'context': result.column_info.context,
				'relevance': result.relevance_score,
				'match_context': result.match_context
			}
		
		return hierarchy

@dataclass
class QueryResult:
	success: bool
	data: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None
	error: Optional[str] = None
	metadata: Optional[Dict[str, Any]] = None
	query_time: float = 0.0
	
@dataclass
class QueryValidation:
	is_valid: bool
	issues: List[str]
	suggested_fixes: Optional[Dict[str, str]] = None
	
@dataclass
class QueryComponents:
	databases: List[str]
	tables: List[str]
	columns: List[str]
	conditions: List[str]
	joins: List[str]
	order_by: List[str]
	group_by: List[str]
	having: List[str]
	limit: Optional[int] = None

class QueryProcessor:
	def __init__(self, llm_interface, context_manager):
		"""Initialize Query Processor with LLM and context manager"""
		self.llm = llm_interface
		self.context_manager = context_manager
		self.logger = logging.getLogger(__name__)
		self.query_history = []
		self.execution_stats = {
			'total_queries': 0,
			'successful_queries': 0,
			'failed_queries': 0,
			'average_execution_time': 0.0
		}

	def process_query(
		self, 
		query: str, 
		context: Dict[str, Any],
		db_connections: Dict[str, sqlite3.Connection]
	) -> QueryResult:
		"""
		Process and execute a SQL query
		
		Args:
			query: SQL query string
			context: Query context information
			db_connections: Dictionary of database connections
			
		Returns:
			QueryResult object containing the results or error
		"""
		try:
			start_time = datetime.now()
			self.execution_stats['total_queries'] += 1

			# Clean and normalize query
			cleaned_query = self._clean_query(query)
			
			# Parse query components
			components = self._parse_query_components(cleaned_query)
			
			# Validate query
			validation = self.validate_query(cleaned_query, context, components)
			if not validation.is_valid:
				self.execution_stats['failed_queries'] += 1
				return QueryResult(
					success=False,
					error="Query validation failed",
					metadata={
						'validation_issues': validation.issues,
						'suggested_fixes': validation.suggested_fixes,
						'query_components': dataclasses.asdict(components)
					}
				)

			# Check database availability
			for db_name in components.databases:
				if db_name not in db_connections:
					self.execution_stats['failed_queries'] += 1
					return QueryResult(
						success=False,
						error=f"Database not available: {db_name}"
					)

			# Execute query on appropriate database(s)
			results = []
			for db_name in components.databases:
				result_df = self._execute_query(cleaned_query, db_connections[db_name])
				results.append(result_df)

			# Combine results if multiple databases
			final_result = pd.concat(results) if len(results) > 1 else results[0]
			
			# Calculate execution time
			query_time = (datetime.now() - start_time).total_seconds()
			
			# Update execution stats
			self.execution_stats['successful_queries'] += 1
			self.execution_stats['average_execution_time'] = (
				(self.execution_stats['average_execution_time'] * 
				 (self.execution_stats['successful_queries'] - 1) + 
				 query_time) / self.execution_stats['successful_queries']
			)
			
			# Update query history
			self._update_query_history(cleaned_query, True, query_time)

			return QueryResult(
				success=True,
				data=final_result,
				metadata={
					'rows': len(final_result),
					'columns': list(final_result.columns),
					'query_components': dataclasses.asdict(components),
					'execution_stats': {
						'time': query_time,
						'databases_used': components.databases
					}
				},
				query_time=query_time
			)

		except Exception as e:
			self.logger.error(f"Error processing query: {str(e)}")
			self.execution_stats['failed_queries'] += 1
			self._update_query_history(query, False, error=str(e))
			return QueryResult(
				success=False,
				error=str(e)
			)

	def validate_query(
		self, 
		query: str, 
		context: Dict[str, Any],
		components: QueryComponents
	) -> QueryValidation:
		"""Validates SQL query against context and components"""
		issues = []
		suggested_fixes = {}

		try:
			# Validate query structure
			structure_validation = self._validate_query_structure(query)
			if not structure_validation['is_valid']:
				issues.extend(structure_validation['issues'])
			
			# Validate databases
			for db in components.databases:
				if not self._validate_database(db, context):
					issues.append(f"Invalid database reference: {db}")
					suggested_fixes[db] = self._suggest_database(db, context)

			# Validate tables
			for table in components.tables:
				if not self._validate_table(table, context):
					issues.append(f"Invalid table reference: {table}")
					suggested_fixes[table] = self._suggest_table(table, context)

			# Validate columns
			for column in components.columns:
				if not self._validate_column(column, context):
					issues.append(f"Invalid column reference: {column}")
					suggested_fixes[column] = self._suggest_column(column, context)

			# Validate joins
			join_validation = self._validate_joins(components.joins, context)
			if not join_validation['is_valid']:
				issues.extend(join_validation['issues'])

			# Validate conditions
			condition_validation = self._validate_conditions(components.conditions, context)
			if not condition_validation['is_valid']:
				issues.extend(condition_validation['issues'])

			return QueryValidation(
				is_valid=len(issues) == 0,
				issues=issues,
				suggested_fixes=suggested_fixes if suggested_fixes else None
			)

		except Exception as e:
			self.logger.error(f"Error validating query: {str(e)}")
			return QueryValidation(
				is_valid=False,
				issues=[str(e)]
			)

	def _execute_query(
		self, 
		query: str, 
		connection: sqlite3.Connection
	) -> pd.DataFrame:
		"""Executes SQL query and returns results as DataFrame"""
		try:
			return pd.read_sql_query(query, connection)
		except Exception as e:
			self.logger.error(f"Error executing query: {str(e)}")
			raise

	def _clean_query(self, query: str) -> str:
		"""Cleans and normalizes SQL query"""
		# Remove extra whitespace
		query = ' '.join(query.split())
		
		# Remove comments
		query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
		query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
		
		# Ensure query ends with semicolon
		if not query.strip().endswith(';'):
			query += ';'
			
		return query

	def _parse_query_components(self, query: str) -> QueryComponents:
		"""Parses query into its components"""
		try:
			parsed = sqlparse.parse(query)[0]
			
			components = QueryComponents(
				databases=[],
				tables=[],
				columns=[],
				conditions=[],
				joins=[],
				order_by=[],
				group_by=[],
				having=[]
			)
			
			# Extract components using sqlparse
			for token in parsed.tokens:
				if isinstance(token, IdentifierList):
					# Handle column list
					for identifier in token.get_identifiers():
						components.columns.append(str(identifier))
						
				elif isinstance(token, Identifier):
					# Handle table names
					if token.get_parent_name():
						components.databases.append(token.get_parent_name())
					components.tables.append(str(token))
					
				elif isinstance(token, Where):
					# Handle WHERE clause
					components.conditions = self._extract_conditions(token)
					
				elif token.ttype is Keyword:
					# Handle other clauses
					if token.value.upper() == 'GROUP BY':
						components.group_by = self._extract_group_by(parsed)
					elif token.value.upper() == 'ORDER BY':
						components.order_by = self._extract_order_by(parsed)
					elif token.value.upper() == 'HAVING':
						components.having = self._extract_having(parsed)
					elif token.value.upper() in ('JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN'):
						components.joins.append(str(token))

			return components
			
		except Exception as e:
			self.logger.error(f"Error parsing query components: {str(e)}")
			raise

	def _validate_query_structure(self, query: str) -> Dict[str, Any]:
		"""Validates basic query structure"""
		issues = []
		
		try:
			# Check for basic SELECT structure
			if not re.match(r'^\s*SELECT\s+.*\s+FROM\s+.*$', query, re.IGNORECASE):
				issues.append("Query must start with SELECT and contain FROM clause")

			# Check for balanced parentheses
			if query.count('(') != query.count(')'):
				issues.append("Unbalanced parentheses in query")

			# Check for invalid characters
			invalid_chars = re.findall(r'[^a-zA-Z0-9_\s\(\)\.\,\*\=\<\>\!\+\-\/\%\;]', query)
			if invalid_chars:
				issues.append(f"Invalid characters found: {set(invalid_chars)}")

			# Check for common SQL injection patterns
			injection_patterns = [
				r'--', r'/\*.*\*/', r';.*$', r'UNION.*SELECT',
				r'DROP.*TABLE', r'DELETE.*FROM', r'INSERT.*INTO'
			]
			for pattern in injection_patterns:
				if re.search(pattern, query, re.IGNORECASE):
					issues.append(f"Potentially unsafe SQL pattern found: {pattern}")

			return {
				'is_valid': len(issues) == 0,
				'issues': issues
			}

		except Exception as e:
			self.logger.error(f"Error validating query structure: {str(e)}")
			return {
				'is_valid': False,
				'issues': [str(e)]
			}

	def _validate_database(self, db_name: str, context: Dict[str, Any]) -> bool:
		"""Validates database reference"""
		return db_name in context

	def _validate_table(self, table_name: str, context: Dict[str, Any]) -> bool:
		"""Validates table reference"""
		for db_info in context.values():
			if table_name in db_info:
				return True
		return False

	def _validate_column(self, column_name: str, context: Dict[str, Any]) -> bool:
		"""Validates column reference"""
		for db_info in context.values():
			for table_info in db_info.values():
				if column_name in table_info:
					return True
		return False

	def _validate_joins(
		self, 
		joins: List[str], 
		context: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Validates join conditions"""
		issues = []
		
		for join in joins:
			# Extract tables and conditions from join
			match = re.match(
				r'.*JOIN\s+(\w+)\s+ON\s+(.*)$',
				join,
				re.IGNORECASE
			)
			if match:
				table, condition = match.groups()
				if not self._validate_table(table, context):
					issues.append(f"Invalid join table: {table}")
				if not self._validate_join_condition(condition, context):
					issues.append(f"Invalid join condition: {condition}")

		return {
			'is_valid': len(issues) == 0,
			'issues': issues
		}

	def _validate_join_condition(
		self, 
		condition: str, 
		context: Dict[str, Any]
	) -> bool:
		"""Validates a join condition"""
		# Implementation for join condition validation
		return True

	def _validate_conditions(
		self, 
		conditions: List[str], 
		context: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Validates WHERE conditions"""
		issues = []
		
		for condition in conditions:
			if isinstance(condition, Comparison):
				left = str(condition.left)
				right = str(condition.right)
				
				# Validate column references
				if not self._validate_column(left, context):
					issues.append(f"Invalid column in condition: {left}")
				
				# Validate value types if possible
				if not self._validate_condition_value(left, right, context):
					issues.append(f"Invalid value type in condition: {right}")

		return {
			'is_valid': len(issues) == 0,
			'issues': issues
		}

	def _validate_condition_value(
		self, 
		column: str, 
		value: str, 
		context: Dict[str, Any]
	) -> bool:
		"""Validates a condition value against column type"""
		# Implementation for condition value validation
		return True

	def _suggest_database(
		self, 
		invalid_db: str, 
		context: Dict[str, Any]
	) -> str:
		"""Suggests similar database names"""
		suggestions = self._get_closest_matches(
			invalid_db,
			list(context.keys())
		)
		return suggestions[0] if suggestions else ""

	def _suggest_table(
		self, 
		invalid_table: str, 
		context: Dict[str, Any]
	) -> str:
		"""Suggests similar table names"""
		all_tables = []
		for db_info in context.values():
			all_tables.extend(db_info.keys())
		
		suggestions = self._get_closest_matches(invalid_table, all_tables)
		return suggestions[0] if suggestions else ""

	def _suggest_column(
		self, 
		invalid_column: str, 
		context: Dict[str, Any]
	) -> str:
		"""Suggests similar column names"""
		all_columns = []
		for db_info in context.values():
			for table_info in db_info.values():
				all_columns.extend(table_info.keys())
		
		suggestions = self._get_closest_matches(invalid_column, all_columns)
		return suggestions[0] if suggestions else ""

	def _get_closest_matches(
		self, 
		target: str, 
		possibilities: List[str]
	) -> List[str]:
		"""Gets closest matching names using Levenshtein distance"""
		from difflib import get_close_matches
		return get_close_matches(target, possibilities, n=3, cutoff=0.6)

	def _extract_conditions(self, where_clause: Where) -> List[str]:
		"""Extracts conditions from WHERE clause"""
		conditions = []
		try:
			for token in where_clause.tokens:
				if isinstance(token, Comparison):
					conditions.append(token)
				elif str(token).upper() in ('AND', 'OR'):
					# Track logical operators
					conditions.append(str(token))
			return conditions
		except Exception as e:
			self.logger.error(f"Error extracting conditions: {str(e)}")
			return []

	def _extract_group_by(self, parsed_query) -> List[str]:
		"""Extracts GROUP BY columns"""
		group_by = []
		in_group_by = False
		
		try:
			for token in parsed_query.tokens:
				if token.ttype is Keyword and token.value.upper() == 'GROUP BY':
					in_group_by = True
					continue
				elif in_group_by:
					if isinstance(token, IdentifierList):
						group_by.extend([str(i) for i in token.get_identifiers()])
					elif isinstance(token, Identifier):
						group_by.append(str(token))
					elif token.ttype is Keyword:
						break
			return group_by
		except Exception as e:
			self.logger.error(f"Error extracting GROUP BY: {str(e)}")
			return []

	def _extract_order_by(self, parsed_query) -> List[str]:
		"""Extracts ORDER BY columns"""
		order_by = []
		in_order_by = False
		
		try:
			for token in parsed_query.tokens:
				if token.ttype is Keyword and token.value.upper() == 'ORDER BY':
					in_order_by = True
					continue
				elif in_order_by:
					if isinstance(token, IdentifierList):
						order_by.extend([str(i) for i in token.get_identifiers()])
					elif isinstance(token, Identifier):
						order_by.append(str(token))
					elif token.ttype is Keyword:
						break
			return order_by
		except Exception as e:
			self.logger.error(f"Error extracting ORDER BY: {str(e)}")
			return []

	def _extract_having(self, parsed_query) -> List[str]:
		"""Extracts HAVING conditions"""
		having = []
		in_having = False
		
		try:
			for token in parsed_query.tokens:
				if token.ttype is Keyword and token.value.upper() == 'HAVING':
					in_having = True
					continue
				elif in_having:
					if isinstance(token, Comparison):
						having.append(str(token))
					elif token.ttype is Keyword:
						break
			return having
		except Exception as e:
			self.logger.error(f"Error extracting HAVING: {str(e)}")
			return []

	def _update_query_history(
		self,
		query: str,
		success: bool,
		execution_time: float = None,
		error: str = None
	):
		"""Updates query execution history"""
		history_entry = {
			'query': query,
			'timestamp': datetime.now(),
			'success': success,
			'execution_time': execution_time
		}
		
		if error:
			history_entry['error'] = error
			
		self.query_history.append(history_entry)
		
		# Keep only last 100 queries
		if len(self.query_history) > 100:
			self.query_history = self.query_history[-100:]

	def get_execution_stats(self) -> Dict[str, Any]:
		"""Returns current execution statistics"""
		return {
			**self.execution_stats,
			'success_rate': (
				self.execution_stats['successful_queries'] /
				self.execution_stats['total_queries']
				if self.execution_stats['total_queries'] > 0 else 0
			)
		}

	def optimize_query(self, query: str, context: Dict[str, Any]) -> str:
		"""Attempts to optimize the query"""
		try:
			components = self._parse_query_components(query)
			optimizations = []
			
			# Check for wildcard select
			if '*' in components.columns:
				specific_columns = self._suggest_specific_columns(components.tables, context)
				optimizations.append({
					'type': 'column_selection',
					'suggestion': f"Specify columns instead of '*': {', '.join(specific_columns)}"
				})
			
			# Check for missing indexes
			missing_indexes = self._check_missing_indexes(components, context)
			if missing_indexes:
				optimizations.append({
					'type': 'missing_indexes',
					'suggestion': f"Consider adding indexes for: {', '.join(missing_indexes)}"
				})
			
			# Check join optimization
			if components.joins:
				join_optimizations = self._optimize_joins(components.joins, context)
				optimizations.extend(join_optimizations)
			
			# Apply optimizations if possible
			optimized_query = self._apply_optimizations(query, optimizations)
			
			return optimized_query
			
		except Exception as e:
			self.logger.error(f"Error optimizing query: {str(e)}")
			return query

	def _suggest_specific_columns(
		self,
		tables: List[str],
		context: Dict[str, Any]
	) -> List[str]:
		"""Suggests specific columns instead of wildcard"""
		suggested_columns = []
		for table in tables:
			for db_info in context.values():
				if table in db_info:
					suggested_columns.extend(
						[f"{table}.{col}" for col in db_info[table].keys()]
					)
		return suggested_columns

	def _check_missing_indexes(
		self,
		components: QueryComponents,
		context: Dict[str, Any]
	) -> List[str]:
		"""Identifies potentially missing indexes"""
		missing_indexes = []
		
		# Check WHERE clause columns
		for condition in components.conditions:
			if isinstance(condition, Comparison):
				column = str(condition.left)
				if self._should_have_index(column, context):
					missing_indexes.append(column)
		
		# Check JOIN conditions
		for join in components.joins:
			join_columns = self._extract_join_columns(join)
			for column in join_columns:
				if self._should_have_index(column, context):
					missing_indexes.append(column)
		
		return missing_indexes

	def _should_have_index(self, column: str, context: Dict[str, Any]) -> bool:
		"""Determines if a column should have an index"""
		# Implementation for index recommendation logic
		return False

	def _optimize_joins(
		self,
		joins: List[str],
		context: Dict[str, Any]
	) -> List[Dict[str, str]]:
		"""Suggests join optimizations"""
		optimizations = []
		
		# Check join order
		if len(joins) > 1:
			optimal_order = self._suggest_join_order(joins, context)
			if optimal_order != joins:
				optimizations.append({
					'type': 'join_order',
					'suggestion': f"Reorder joins: {', '.join(optimal_order)}"
				})
		
		# Check join types
		for join in joins:
			suggested_type = self._suggest_join_type(join, context)
			if suggested_type:
				optimizations.append({
					'type': 'join_type',
					'suggestion': f"Consider using {suggested_type} for {join}"
				})
		
		return optimizations

	def _suggest_join_order(
		self,
		joins: List[str],
		context: Dict[str, Any]
	) -> List[str]:
		"""Suggests optimal join order"""
		# Implementation for join order optimization
		return joins

	def _suggest_join_type(
		self,
		join: str,
		context: Dict[str, Any]
	) -> Optional[str]:
		"""Suggests optimal join type"""
		# Implementation for join type optimization
		return None

	def _apply_optimizations(
		self,
		query: str,
		optimizations: List[Dict[str, str]]
	) -> str:
		"""Applies suggested optimizations to the query"""
		optimized_query = query
		
		for opt in optimizations:
			if opt['type'] == 'column_selection':
				optimized_query = self._apply_column_selection(
					optimized_query,
					opt['suggestion']
				)
			elif opt['type'] == 'join_order':
				optimized_query = self._apply_join_order(
					optimized_query,
					opt['suggestion']
				)
			elif opt['type'] == 'join_type':
				optimized_query = self._apply_join_type(
					optimized_query,
					opt['suggestion']
				)
		
		return optimized_query

	def _apply_column_selection(
		self, 
		query: str, 
		columns: str
	) -> str:
		"""Applies column selection optimization"""
		return re.sub(r'SELECT\s+\*', f'SELECT {columns}', query)

	def _apply_join_order(
		self,
		query: str,
		join_order: str
	) -> str:
		"""Applies join order optimization"""
		# Implementation for applying join order
		return query

	def _apply_join_type(
		self,
		query: str,
		join_type: str
	) -> str:
		"""Applies join type optimization"""
		# Implementation for applying join type
		return query

class ResponseFormatter:
	def __init__(self):
		self.logger = logging.getLogger(__name__)

	def format_response(
		self,
		response: Union[str, Dict[str, Any], LLMResponse],
		format_type: str = "default"
	) -> str:
		"""Formats response based on type and content"""
		try:
			# If response is LLMResponse, get content
			if isinstance(response, LLMResponse):
				return response.content
			
			# If response is a string, return it directly
			if isinstance(response, str):
				# Check if the string is a JSON representation
				try:
					parsed = json.loads(response)
					if "intent" in parsed:
						return "Hello! I'm your data analysis assistant. How can I help you today?"
					return response
				except json.JSONDecodeError:
					return response
			
			# For other types, convert to string
			return str(response)
			
		except Exception as e:
			self.logger.error(f"Error formatting response: {str(e)}")
			return "I apologize, but I encountered an error formatting the response."

	def _format_query_result(self, result: QueryResult) -> str:
		"""Formats query execution result"""
		if not result.success:
			return self._format_error_response(result.error, result.metadata)

		try:
			output_parts = []
			
			# Add execution metadata
			output_parts.append(
				f"Query executed successfully in {result.query_time:.2f} seconds"
			)
			output_parts.append(f"Returned {result.metadata['rows']} rows\n")

			# Format data
			if isinstance(result.data, pd.DataFrame):
				# Convert DataFrame to string with proper formatting
				output_parts.append(
					result.data.to_string(
						index=False,
						max_rows=100,
						max_cols=None,
						max_colwidth=20
					)
				)
			else:
				output_parts.append(json.dumps(result.data, indent=2))

			return "\n".join(output_parts)

		except Exception as e:
			self.logger.error(f"Error formatting query result: {str(e)}")
			return str(result.data)

	def format_hierarchical_response(
		self,
		data: Dict[str, Any],
		indent: int = 0
	) -> str:
		"""Formats response in a hierarchical structure"""
		try:
			output_parts = []
			
			for db_name, db_info in data.items():
				output_parts.append(f"\nDatabase: {db_name}")
				
				for table_name, table_info in db_info.items():
					output_parts.append(f"{' ' * 2}Table: {table_name}")
					
					for column_name, column_info in table_info.items():
						output_parts.append(
							f"{' ' * 4}Column: {column_name} "
							f"({column_info.get('data_type', 'UNKNOWN')})"
						)
						if 'context' in column_info:
							output_parts.append(
								f"{' ' * 6}{column_info['context']}"
							)

			return "\n".join(output_parts)

		except Exception as e:
			self.logger.error(f"Error formatting hierarchical response: {str(e)}")
			return str(data)

	def format_discovery_response(
		self,
		discoveries: List[Dict[str, Any]]
	) -> str:
		"""Formats data discovery results"""
		try:
			output_parts = ["\nRelevant Data Discoveries:"]
			
			for disc in discoveries:
				output_parts.append(f"\n• {disc['attribute']}")
				output_parts.append(f"  Location: {disc['database']}.{disc['table']}")
				output_parts.append(f"  Type: {disc['data_type']}")
				if 'description' in disc:
					output_parts.append(f"  Description: {disc['description']}")
				if 'relevance' in disc:
					output_parts.append(
						f"  Relevance: {disc['relevance']*100:.1f}%"
					)

			return "\n".join(output_parts)

		except Exception as e:
			self.logger.error(f"Error formatting discovery response: {str(e)}")
			return str(discoveries)

	def _format_error_response(
		self,
		error: str,
		metadata: Optional[Dict[str, Any]] = None
	) -> str:
		"""Formats error responses"""
		output_parts = [f"Error: {error}"]
		
		if metadata:
			if 'validation_issues' in metadata:
				output_parts.append("\nValidation Issues:")
				for issue in metadata['validation_issues']:
					output_parts.append(f"• {issue}")
			
			if 'suggested_fixes' in metadata:
				output_parts.append("\nSuggested Fixes:")
				for original, suggestion in metadata['suggested_fixes'].items():
					output_parts.append(f"• {original} → {suggestion}")

		return "\n".join(output_parts)

	def _format_dict_response(
		self,
		data: Dict[str, Any],
		format_type: str
	) -> str:
		"""Formats dictionary responses"""
		if format_type == "hierarchical":
			return self.format_hierarchical_response(data)
		else:
			return json.dumps(data, indent=2)

	def _format_text_response(self, text: str) -> str:
		"""Formats text responses"""
		return text.strip()

def load_context(context_file: str = 'compressed_context.json') -> Dict[str, Any]:
	"""
	Load context information from JSON file
	
	Args:
		context_file: Path to the context JSON file
		
	Returns:
		Dictionary containing database schema and context information
	"""
	try:
		with open(context_file, 'r') as f:
			context = json.load(f)
			
		# Debug print
		print("\nLoaded Context Structure:")
		for db_name, db_details in context.items():
			print(f"\nDatabase: {db_name}")
			for table_name, table_details in db_details.items():
				print(f"  Table: {table_name}")
				print(f"  Columns: {len(list(table_details.keys()))}")
		
		return context
		
	except FileNotFoundError:
		logger.error(f"Context file {context_file} not found")
		print(f"Error: Could not find context file {context_file}")
		return {}
	except json.JSONDecodeError:
		logger.error(f"Invalid JSON format in {context_file}")
		print(f"Error: Invalid JSON format in {context_file}")
		return {}
	except Exception as e:
		logger.error(f"Error loading context: {str(e)}")
		print(f"Error loading context: {str(e)}")
		return {}

class DataCopilot:
	def __init__(self):
		try:
			# Load context first
			self.context_info = load_context()
			
			# Initialize components with proper order
			self.llm_interface = LLMInterface()
			
			# Create context manager with concept extraction
			self.context_manager = ContextManager(self.context_info)
			
			# Initialize other processors with the enhanced context manager
			self.intent_processor = IntentProcessor(self.llm_interface, self.context_manager)
			self.query_processor = QueryProcessor(self.llm_interface, self.context_manager)
			self.response_formatter = ResponseFormatter()
			
			# Initial concept clustering
			logger.info("Initializing concept clusters...")
			self.context_manager._initialize_concept_clusters()
			logger.info("Concept clusters initialized")
			
		except Exception as e:
			logger.error(f"Error initializing DataCopilot: {str(e)}")
			raise

	@tool(
		name="process_input",
		description="Process user input and generate appropriate response"
	)
	def process_input(self, user_input: str) -> str:
		try:
			# Detect intent
			intent = self.intent_processor.detect_intent(user_input)
			
			# Process based on intent
			raw_response = self.intent_processor.process_intent(intent, user_input)
			
			# Format response
			formatted_response = self.response_formatter.format_response(
				raw_response,
				str(intent.type.value)
			)
			
			return formatted_response
			
		except Exception as e:
			logger.error(f"Error processing input: {str(e)}")
			return f"I apologize, but I encountered an error: {str(e)}"

	def run_interactive(self):
		logger.info("Starting interactive session...")
		print("\nData Copilot >")
		print("    Welcome! I'm J, your AI Copilot for data discovery and data science.")
		print("    Type 'exit' to end our conversation.\n")
		
		while True:
			try:
				user_input = input("Query > ").strip()
				
				if user_input.lower() == 'exit':
					break
					
				response = self.process_input(user_input)
				print(f"\nData Copilot > {response}\n")
				
			except Exception as e:
				print(f"\nData Copilot > An error occurred: {str(e)}\n")
				logger.error(f"Error in interaction: {str(e)}")

if __name__ == "__main__":
	# Set up logging configuration
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	)
	
	try:
		copilot = DataCopilot()
		copilot.run_interactive()
	except Exception as e:
		logger.error(f"Fatal error: {str(e)}")
		print(f"Fatal error occurred: {str(e)}")
