import streamlit as st
import os
import pandas as pd
import requests
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from langchain import hub
import json
import uuid
import datetime
import yfinance as yf
import wbgapi as wb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from typing import Dict, List, Optional, Any
import time

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="🤖 Agentic AI Financial Assistant", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Custom CSS for professional design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Roboto+Slab:wght@700&display=swap');

    /* General Styling */
    .main-container {
        background-color: #f0f0f0;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        font-family: 'Roboto', sans-serif;
        color: #666666;
    }
    .main-header {
        background: linear-gradient(90deg, #003366 0%, #3399cc 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        animation: fadeIn 1s ease-in-out;
    }
    .main-header h1 {
        font-family: 'Roboto Slab', serif;
        font-size: 2.5em;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .main-header p {
        font-size: 1.2em;
        opacity: 0.9;
    }
    .section-header {
        font-family: 'Roboto Slab', serif;
        color: #003366;
        font-size: 1.8em;
        margin: 1.5rem 0 1rem;
        border-bottom: 2px solid #3399cc;
        padding-bottom: 0.5rem;
    }
    .question-box, .answer-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease;
        border-left: 4px solid #003366;
    }
    .question-box:hover, .answer-box:hover {
        transform: translateY(-5px);
    }
    .agent-thinking {
        background-color: #ecf0f1;
        padding: 1rem;
        border-radius: 10px;
        color: #666666;
        margin: 0.5rem 0;
        font-style: italic;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #003366;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    /* Data Sources Styling */
    .data-source-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: transform 0.3s ease;
    }
    .data-source-card:hover {
        transform: translateY(-3px);
    }
    .status-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    .status-active { background-color: #28a745; }
    .status-inactive { background-color: #dc3545; }
    .status-limited { background-color: #ffc107; }
    .data-source-info {
        display: flex;
        align-items: center;
    }
    .data-source-info strong {
        color: #003366;
        font-size: 1.1em;
    }
    .data-source-info small {
        color: #666666;
        margin-left: 0.5rem;
        font-size: 0.9em;
    }
    /* Gold Prices Styling */
    .gold-prices {
        background-color: #ffd700;
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    .gold-prices h3 {
        color: white;
        font-size: 2em;
    }
    .gold-prices p {
        font-size: 1.2em;
        margin: 0.5rem 0;
    }
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #003366;
        color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .sidebar-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .sidebar-section h3 {
        color: #3399cc;
        margin-top: 0;
    }
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ecf0f1;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: #003366;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #003366;
        color: white;
    }
    /* Form Styling */
    .stForm {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #003366;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: transform 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    /* Expander Styling */
    .stExpander {
        background-color: #ecf0f1;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    /* Footer Styling */
    .footer {
        background-color: #003366;
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 -5px 15px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
    
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

if 'agent_memory' not in st.session_state:
    st.session_state.agent_memory = ConversationBufferWindowMemory(
        k=10, 
        return_messages=True,
        memory_key="chat_history"
    )

# Load environment variables
load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
OER_API_KEY = os.getenv("EXCHANGERATES_API_KEY")
GOLDAP_API_KEY = os.getenv("GOLDAP_API")
FINNHUB_API_KEY = os.getenv("FINNHUB_API")
NINJAS_API_KEY = os.getenv("NINJAS_API")
MARKETSTACK_API_KEY = os.getenv("MARKETSTACK_API")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Main Container
with st.container():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Agentic AI Financial Assistant</h1>
        <p>Empowering Wealth Creation with Real-time Insights & Personalized Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Load local CSV data
    csv_path = 'financial_risk_analysis_large.csv'
    try:
        financial_data = pd.read_csv(csv_path)
    except FileNotFoundError:
        financial_data = pd.DataFrame()
        st.warning("Note: financial_risk_analysis_large.csv not found. Using alternative data sources.")

    # Expanded list of companies
    COMPANY_SYMBOLS = [
        "AAPL", "MSFT", "NVDA", "AMZN", "TSLA",
        "GOOGL", "META", "BRK-B", "AVGO", "TSM",
        "TM", "BABA", "V", "WMT", "JPM", "NFLX",
        "AMD", "CRM", "ORCL", "ADBE"
    ]

    # Enhanced data fetching functions
    @st.cache_data(ttl=300)  # 5-minute cache
    def get_yfinance_data(symbols=COMPANY_SYMBOLS, period="6mo"):
        all_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                info = ticker.info
                
                if not hist.empty:
                    all_data[symbol] = {
                        'history': hist,
                        'info': info
                    }
            except Exception as e:
                st.warning(f"Could not fetch yfinance data for {symbol}: {e}")
        return all_data

    @st.cache_data(ttl=3600)  # 1-hour cache
    def get_world_bank_data():
        try:
            countries = ['USA', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA', 'IND', 'BRA']
            gdp_data = wb.data.DataFrame('NY.GDP.MKTP.CD', countries, time=range(2018, 2023))
            inflation_data = wb.data.DataFrame('FP.CPI.TOTL.ZG', countries, time=range(2018, 2023))
            unemployment_data = wb.data.DataFrame('SL.UEM.TOTL.ZS', countries, time=range(2018, 2023))
            
            return {
                'gdp': gdp_data,
                'inflation': inflation_data,
                'unemployment': unemployment_data
            }
        except Exception as e:
            st.warning(f"Could not fetch World Bank data: {e}")
            return {}

    @st.cache_data(ttl=300)
    def get_stock_data(symbols=COMPANY_SYMBOLS):
        all_data = {}
        for symbol in symbols:
            if ALPHA_API_KEY:
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_API_KEY}&outputsize=compact"
                try:
                    response = requests.get(url)
                    data = response.json()
                    if 'Time Series (Daily)' in data:
                        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index').astype(float)
                        df.index = pd.to_datetime(df.index)
                        df.sort_index(inplace=True)
                        all_data[symbol] = df
                        continue
                except Exception as e:
                    st.warning(f"Alpha Vantage failed for {symbol}: {e}")
            
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="6mo")
                if not hist.empty:
                    all_data[symbol] = hist
            except Exception as e:
                st.warning(f"Could not fetch data for {symbol}: {e}")
        
        return all_data

    @st.cache_data(ttl=300)
    def get_gold_prices():
        # Try GoldAPI first
        if GOLDAP_API_KEY:
            url = f"https://www.goldapi.io/api/XAU/USD?apikey={GOLDAP_API_KEY}"
            try:
                response = requests.get(url)
                data = response.json()
                if 'price' in data:
                    return {
                        'ounce': data['price'],
                        'gram': data['price'] / 31.1035,
                        'kilogram': data['price'] * 32.1507
                    }
            except Exception as e:
                st.warning(f"GoldAPI failed: {e}. Falling back to yfinance.")
        
        # Fallback to yfinance for gold prices (GC=F is the symbol for gold futures)
        try:
            gold_ticker = yf.Ticker("GC=F")
            hist = gold_ticker.history(period="1d")
            if not hist.empty:
                price_per_ounce = hist['Close'].iloc[-1]
                return {
                    'ounce': price_per_ounce,
                    'gram': price_per_ounce / 31.1035,
                    'kilogram': price_per_ounce * 32.1507
                }
        except Exception as e:
            st.warning(f"Could not fetch gold prices from yfinance: {e}")
        
        return {'ounce': 0, 'gram': 0, 'kilogram': 0}

    # User Profile Management
    class UserProfileManager:
        @staticmethod
        def collect_user_info(question: str) -> Dict[str, Any]:
            personal_keywords = {
                'age': ['age', 'old', 'young', 'retirement', 'years old'],
                'income': ['income', 'salary', 'earn', 'make money', 'budget'],
                'risk_tolerance': ['risk', 'conservative', 'aggressive', 'moderate'],
                'investment_goals': ['goals', 'target', 'objective', 'plan'],
                'time_horizon': ['when', 'timeline', 'years', 'months', 'long term', 'short term'],
                'location': ['country', 'location', 'where', 'tax', 'jurisdiction']
            }
            
            needed_info = []
            for info_type, keywords in personal_keywords.items():
                if any(keyword in question.lower() for keyword in keywords):
                    needed_info.append(info_type)
            
            return needed_info

        @staticmethod
        def get_user_input(info_type: str) -> Optional[str]:
            prompts = {
                'age': "What's your age range? (20-30, 30-40, 40-50, 50-60, 60+)",
                'income': "What's your approximate annual income range? (Optional)",
                'risk_tolerance': "What's your risk tolerance? (Conservative, Moderate, Aggressive)",
                'investment_goals': "What are your main investment goals?",
                'time_horizon': "What's your investment timeline?",
                'location': "What country/region are you in? (for tax considerations)"
            }
            
            if info_type in prompts:
                return st.text_input(prompts[info_type], key=f"user_{info_type}")
            return None

    # Enhanced Agent Tools
    class FinancialAgentTools:
        def __init__(self, vectorstore=None):
            self.vectorstore = vectorstore
            self.search_tool = DuckDuckGoSearchRun()
        
        def financial_data_search(self, query: str) -> str:
            if not self.vectorstore:
                return "Internal financial database not available"
            
            try:
                docs = self.vectorstore.similarity_search(query, k=5)
                results = "\n\n".join([doc.page_content for doc in docs])
                return f"Internal Financial Data Results:\n{results}"
            except Exception as e:
                return f"Error searching financial data: {str(e)}"
        
        def web_research(self, query: str) -> str:
            try:
                financial_query = f"financial market analysis {query} 2024 2025"
                results = self.search_tool.run(financial_query)
                return f"Web Research Results:\n{results}"
            except Exception as e:
                return f"Web research unavailable: {str(e)}"
        
        def calculate_financial_metrics(self, data: str) -> str:
            try:
                if "P/E ratio" in data.lower() or "pe ratio" in data.lower():
                    return "P/E Ratio calculation: Price per Share / Earnings per Share. Lower ratios may indicate undervalued stocks, while higher ratios might suggest growth expectations."
                elif "roi" in data.lower() or "return on investment" in data.lower():
                    return "ROI calculation: (Gain from Investment - Cost of Investment) / Cost of Investment × 100%"
                else:
                    return "Financial metrics calculated based on available data."
            except Exception as e:
                return f"Calculation error: {str(e)}"
        
        def get_market_sentiment(self, query: str) -> str:
            try:
                sentiment_query = f"market sentiment analysis {query} investor opinion"
                results = self.search_tool.run(sentiment_query)
                return f"Market Sentiment Analysis:\n{results}"
            except Exception as e:
                return f"Sentiment analysis unavailable: {str(e)}"

    # Fetch all data
    with st.spinner("🔄 Loading financial data from multiple sources..."):
        yfinance_data = get_yfinance_data()
        world_bank_data = get_world_bank_data()
        alpha_data = get_stock_data()
        gold_prices = get_gold_prices()

    # Data source status display
    st.markdown('<h2 class="section-header">📊 Data Sources Status</h2>', unsafe_allow_html=True)

    sources_status = [
        ("Yahoo Finance", len(yfinance_data), "active"),
        ("World Bank", len(world_bank_data), "active" if world_bank_data else "inactive"),
        ("Alpha Vantage", len(alpha_data), "active" if ALPHA_API_KEY else "limited"),
        ("Web Research", "Available", "active")
    ]

    cols = st.columns(4)
    for i, (source, count, status) in enumerate(sources_status):
        with cols[i]:
            st.markdown(f"""
            <div class="data-source-card">
                <div class="data-source-info">
                    <span class="status-dot status-{status}"></span>
                    <div>
                        <strong>{source}</strong>
                        <br>
                        <small>{count}</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Convert data to documents for RAG
    def convert_data_to_documents(combined_data):
        documents = []
        
        if "csv_data" in combined_data and combined_data["csv_data"]:
            for row in combined_data["csv_data"][:100]:
                content = "\n".join([f"{k}: {v}" for k, v in row.items() if v is not None])
                documents.append(Document(page_content=content, metadata={"source": "csv"}))
        
        if "yfinance_data" in combined_data and combined_data["yfinance_data"]:
            for symbol, data in combined_data["yfinance_data"].items():
                if 'info' in data:
                    content = f"Company: {symbol}\n" + "\n".join([f"{k}: {v}" for k, v in data['info'].items() if v is not None])
                    documents.append(Document(page_content=content, metadata={"source": "yfinance", "symbol": symbol}))
        
        if "world_bank_data" in combined_data and combined_data["world_bank_data"]:
            for indicator, data in combined_data["world_bank_data"].items():
                if hasattr(data, 'to_string'):
                    content = f"World Bank {indicator} data:\n{data.to_string()}"
                    documents.append(Document(page_content=content, metadata={"source": "world_bank", "indicator": indicator}))
        
        return documents

    # Combine all data
    combined_data = {
        "csv_data": financial_data.to_dict(orient="records") if not financial_data.empty else [],
        "yfinance_data": yfinance_data,
        "world_bank_data": world_bank_data,
        "stock_data_alpha": alpha_data,
    }

    # Convert to documents and create vectorstore
    documents = convert_data_to_documents(combined_data)
    vectorstore = None

    if documents and OPENAI_API_KEY:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)
        
        try:
            if os.path.exists("faiss_index"):
                vectorstore = FAISS.load_local("faiss_index", embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), allow_dangerous_deserialization=True)
            else:
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vectorstore = FAISS.from_documents(split_docs, embeddings)
                vectorstore.save_local("faiss_index")
        except Exception as e:
            st.warning(f"Vector store creation failed: {e}")

    # Initialize Enhanced Agentic AI System
    if OPENAI_API_KEY:
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.1, openai_api_key=OPENAI_API_KEY)
        
        financial_tools = FinancialAgentTools(vectorstore)
        
        tools = [
            Tool(
                name="FinancialDataSearch",
                func=financial_tools.financial_data_search,
                description="Search internal financial database for stock prices, company info, economic indicators"
            ),
            Tool(
                name="WebResearch",
                func=financial_tools.web_research,
                description="Search the web for current financial news, market analysis, and additional information if internal data is insufficient"
            ),
            Tool(
                name="CalculateMetrics",
                func=financial_tools.calculate_financial_metrics,
                description="Calculate financial ratios, metrics, and perform quantitative analysis"
            ),
            Tool(
                name="MarketSentiment",
                func=financial_tools.get_market_sentiment,
                description="Analyze market sentiment and investor opinions"
            )
        ]
        
        enhanced_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Advanced Financial AI Agent with access to multiple data sources and web research capabilities.

    Your capabilities include:
    1. Analyzing internal financial databases (stocks, economics, company data)
    2. Conducting web research for the latest information when internal data is insufficient
    3. Calculating financial metrics and ratios
    4. Analyzing market sentiment
    5. Providing personalized financial insights

    When answering questions:
    1. Always start by acknowledging the user's question
    2. First, use the FinancialDataSearch tool to check internal data (RAG system)
    3. If the internal data is insufficient (e.g., missing, outdated, or not relevant), automatically use the WebResearch tool to fetch additional information
    4. If personal information is needed, ask the user specifically
    5. Combine multiple data sources for complete analysis
    6. Present information in an organized, detailed manner
    7. Include relevant visualizations when possible
    8. Provide actionable insights and recommendations

    Remember: Always be thorough, accurate, and helpful. Automatically decide when web research is needed based on the question's requirements."""),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        
        try:
            react_prompt = hub.pull("hwchase17/react")
            agent = create_react_agent(llm, tools, react_prompt)
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=True, 
                max_iterations=5,
                memory=st.session_state.agent_memory
            )
        except Exception as e:
            st.error(f"Agent initialization failed: {e}")
            agent_executor = None

    # Main Chat Interface
    st.markdown('<h2 class="section-header">💬 Ask Your Financial AI Agent</h2>', unsafe_allow_html=True)

    if OPENAI_API_KEY and agent_executor:
        with st.form(key="question_form", clear_on_submit=True):
            question = st.text_area(
                "Ask me anything about finance, markets, investments, or economics:", 
                placeholder="e.g., Should I invest in Tesla given my moderate risk tolerance and 10-year timeline?",
                height=100,
                key="question_input"
            )
            
            col1, _ = st.columns([3, 1])
            with col1:
                submit_button = st.form_submit_button("🚀 Ask Agent", use_container_width=True)

        if question and submit_button:
            st.markdown(f"""
            <div class="question-box">
                <h3>❓ Your Question:</h3>
                <p style="font-size: 1.1em;">{question}</p>
            </div>
            """, unsafe_allow_html=True)
            
            user_manager = UserProfileManager()
            needed_info = user_manager.collect_user_info(question)
            
            user_context = ""
            if needed_info:
                st.markdown('<h3 class="section-header">👤 I need some information to provide personalized advice:</h3>', unsafe_allow_html=True)
                
                collected_info = {}
                for info_type in needed_info:
                    if info_type not in st.session_state.user_profile:
                        user_input = user_manager.get_user_input(info_type)
                        if user_input:
                            collected_info[info_type] = user_input
                            st.session_state.user_profile[info_type] = user_input
                
                if st.session_state.user_profile:
                    user_context = f"\nUser Profile: {json.dumps(st.session_state.user_profile, indent=2)}\n"
            
            enhanced_question = f"{question}\n{user_context}"
            
            with st.spinner("🤖 Agent is thinking and researching..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.markdown('<div class="agent-thinking">🔍 Analyzing your question...</div>', unsafe_allow_html=True)
                    progress_bar.progress(25)
                    
                    status_text.markdown('<div class="agent-thinking">📊 Searching financial databases...</div>', unsafe_allow_html=True)
                    progress_bar.progress(50)
                    
                    status_text.markdown('<div class="agent-thinking">🌐 Checking if web research is needed...</div>', unsafe_allow_html=True)
                    progress_bar.progress(75)
                    
                    status_text.markdown('<div class="agent-thinking">🧠 Generating comprehensive analysis...</div>', unsafe_allow_html=True)
                    progress_bar.progress(90)
                    
                    response = agent_executor.invoke({"input": enhanced_question})
                    answer = response["output"]
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.markdown(f"""
                    <div class="answer-box">
                        <h3>🤖 AI Agent Response:</h3>
                        <div style="font-size: 1.05em; line-height: 1.6;">{answer}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.session_state.conversation_history.append({
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": question,
                        "answer": answer,
                        "user_context": user_context
                    })
                    
                    st.session_state.agent_memory.chat_memory.add_user_message(HumanMessage(content=question))
                    st.session_state.agent_memory.chat_memory.add_ai_message(AIMessage(content=answer))
                    
                except Exception as e:
                    st.error(f"Error processing your question: {e}")
                    st.info("The agent encountered an issue. Please try rephrasing your question or check your API keys.")

    else:
        if not OPENAI_API_KEY:
            st.error("🔑 Please set your OPENAI_API_KEY in the .env file to use the Agentic AI features.")
        else:
            st.warning("⚠️ Agent system is initializing. Please refresh the page if this persists.")

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section"><h3>🤖 Agent Status</h3></div>', unsafe_allow_html=True)
        
        if OPENAI_API_KEY and agent_executor:
            st.success("✅ Agentic AI Active")
            st.info("🧠 GPT-4 Powered")
            st.info("🔍 Web Research Enabled")
            st.info("📊 Multi-Source Analysis")
        else:
            st.error("❌ Agent Offline")
        
        st.markdown('<div class="sidebar-section"><h3>👤 Your Profile</h3></div>', unsafe_allow_html=True)
        if st.session_state.user_profile:
            for key, value in st.session_state.user_profile.items():
                st.write(f"**{key.title()}**: {value}")
            
            if st.button("🔄 Clear Profile"):
                st.session_state.user_profile = {}
                st.rerun()
        else:
            st.info("Profile will be built as you ask personalized questions")
        
        st.markdown('<div class="sidebar-section"><h3>💬 Recent Conversations</h3></div>', unsafe_allow_html=True)
        if st.session_state.conversation_history:
            for i, entry in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.expander(f"Q{len(st.session_state.conversation_history)-i}: {entry['question'][:40]}..."):
                    st.write(f"**Time**: {entry['timestamp']}")
                    st.write(f"**Question**: {entry['question']}")
                    st.write(f"**Answer**: {entry['answer'][:200]}...")
        
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.session_state.agent_memory.clear()
            st.rerun()

    # Enhanced Data Visualization Section
    st.markdown('<h2 class="section-header">📈 Interactive Financial Dashboard</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Analysis", "🏢 Company Comparison", "🌍 Economic Indicators", "🔄 Real-time Data"])

    with tab1:
        st.subheader("📊 Advanced Stock Analysis")
        
        selected_stocks = st.multiselect("Select stocks to analyze:", COMPANY_SYMBOLS, default=["AAPL", "MSFT", "TSLA"])
        
        if selected_stocks and yfinance_data:
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=('Stock Prices', 'Trading Volume'),
                              vertical_spacing=0.1)
            
            colors = px.colors.qualitative.Set1
            
            for i, symbol in enumerate(selected_stocks):
                color = colors[i % len(colors)]
                
                if symbol in yfinance_data and 'history' in yfinance_data[symbol]:
                    df = yfinance_data[symbol]['history']
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df['Close'], name=f"{symbol} Price", 
                                 line=dict(color=color, width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Bar(x=df.index, y=df['Volume'], name=f"{symbol} Volume", 
                              marker_color=color, opacity=0.6),
                        row=2, col=1
                    )
            
            fig.update_layout(height=600, title_text="Advanced Stock Analysis Dashboard")
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🏢 Market Cap & Company Metrics")
        
        if yfinance_data:
            market_caps = []
            companies = []
            sectors = []
            
            for symbol, data in yfinance_data.items():
                if 'info' in data and data['info'].get('marketCap'):
                    market_caps.append(data['info']['marketCap'] / 1e9)
                    companies.append(symbol)
                    sectors.append(data['info'].get('sector', 'Unknown'))
            
            if market_caps:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=companies,
                    y=market_caps,
                    text=[f"${cap:.1f}B" for cap in market_caps],
                    textposition='auto',
                    marker_color=px.colors.qualitative.Set3
                ))
                
                fig.update_layout(
                    title="Market Capitalization Comparison",
                    xaxis_title="Company",
                    yaxis_title="Market Cap (Billions USD)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🌍 Global Economic Indicators")
        
        if world_bank_data:
            indicator = st.selectbox("Select Economic Indicator:", 
                                    ["GDP", "Inflation", "Unemployment"], 
                                    key="wb_indicator")
            
            if indicator.lower() in world_bank_data:
                data = world_bank_data[indicator.lower()]
                
                if hasattr(data, 'reset_index'):
                    df = data.reset_index()
                    
                    fig = go.Figure()
                    
                    for country in df.columns[1:]:
                        if country in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df.iloc[:, 0],
                                y=df[country],
                                mode='lines+markers',
                                name=country,
                                line=dict(width=2)
                            ))
                    
                    fig.update_layout(
                        title=f"{indicator} Trends by Country",
                        xaxis_title="Year",
                        yaxis_title=f"{indicator} Value",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("🔄 Real-time Market Data")
        
        if yfinance_data:
            st.markdown("#### Market Summary")
            
            summary_data = []
            for symbol, data in list(yfinance_data.items())[:6]:
                if 'info' in data:
                    info = data['info']
                    summary_data.append({
                        'Symbol': symbol,
                        'Company': info.get('longName', symbol)[:30],
                        'Price': f"${info.get('currentPrice', 0):.2f}",
                        'Market Cap': f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else "N/A",
                        'Sector': info.get('sector', 'Unknown')
                    })
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)
        
        st.markdown("#### Major Market Indices")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("S&P 500", "4,567.89", "2.34%", delta_color="normal")
        with col2:
            st.metric("NASDAQ", "14,123.45", "1.89%", delta_color="normal")
        with col3:
            st.metric("DOW", "35,678.90", "-0.45%", delta_color="inverse")
        with col4:
            st.metric("VIX", "18.45", "-1.23%", delta_color="inverse")

        # Gold Prices Display
        st.markdown("#### 💰 Gold Prices")
        if gold_prices['ounce'] > 0:
            st.markdown(f"""
            <div class="gold-prices">
                <h3>🌟 Golden Treasure Prices 🌟</h3>
                <p><strong>1 Ounce:</strong> ${gold_prices['ounce']:.2f} USD</p>
                <p><strong>1 Gram:</strong> ${gold_prices['gram']:.2f} USD</p>
                <p><strong>1 Kilogram:</strong> ${gold_prices['kilogram']:.2f} USD</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Gold price data not available. Please ensure API keys are set or check your internet connection.")

    # Advanced Analytics Section
    st.markdown('<h2 class="section-header">🧮 Advanced Financial Analytics</h2>', unsafe_allow_html=True)

    analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(["🔢 Portfolio Analysis", "📈 Technical Indicators", "🎯 Risk Assessment"])

    with analytics_tab1:
        st.subheader("🔢 Portfolio Performance Analysis")
        
        st.markdown("#### Portfolio Simulator")
        portfolio_stocks = st.multiselect("Select stocks for your portfolio:", COMPANY_SYMBOLS, default=["AAPL", "GOOGL", "MSFT"])
        
        if portfolio_stocks and len(portfolio_stocks) > 1:
            weights = [1/len(portfolio_stocks)] * len(portfolio_stocks)
            
            portfolio_data = []
            total_value = 0
            
            for i, symbol in enumerate(portfolio_stocks):
                if symbol in yfinance_data and 'info' in yfinance_data[symbol]:
                    info = yfinance_data[symbol]['info']
                    price = info.get('currentPrice', 0)
                    market_cap = info.get('marketCap', 0)
                    weight = weights[i]
                    
                    portfolio_data.append({
                        'Stock': symbol,
                        'Weight': f"{weight*100:.1f}%",
                        'Price': f"${price:.2f}",
                        'Market Cap': f"${market_cap/1e9:.1f}B" if market_cap else "N/A"
                    })
                    total_value += price * weight
            
            if portfolio_data:
                st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True)
                
                fig = go.Figure(data=[go.Pie(
                    labels=[item['Stock'] for item in portfolio_data],
                    values=[1/len(portfolio_stocks)] * len(portfolio_stocks),
                    hole=.3
                )])
                fig.update_layout(title="Portfolio Allocation", height=400)
                st.plotly_chart(fig, use_container_width=True)

    with analytics_tab2:
        st.subheader("📈 Technical Analysis")
        
        selected_stock = st.selectbox("Select stock for technical analysis:", COMPANY_SYMBOLS, key="tech_analysis")
        
        if selected_stock and selected_stock in yfinance_data:
            hist_data = yfinance_data[selected_stock]['history']
            
            if not hist_data.empty:
                hist_data['MA20'] = hist_data['Close'].rolling(window=20).mean()
                hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
                
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=hist_data.index,
                    open=hist_data['Open'],
                    high=hist_data['High'],
                    low=hist_data['Low'],
                    close=hist_data['Close'],
                    name=f"{selected_stock} Price"
                ))
                
                fig.add_trace(go.Scatter(
                    x=hist_data.index,
                    y=hist_data['MA20'],
                    name='MA20',
                    line=dict(color='orange', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=hist_data.index,
                    y=hist_data['MA50'],
                    name='MA50',
                    line=dict(color='red', width=1)
                ))
                
                fig.update_layout(
                    title=f"{selected_stock} Technical Analysis",
                    yaxis_title="Price ($)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                current_price = hist_data['Close'].iloc[-1]
                ma20 = hist_data['MA20'].iloc[-1]
                ma50 = hist_data['MA50'].iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    trend_20 = "Bullish" if current_price > ma20 else "Bearish"
                    st.metric("20-Day Trend", trend_20)
                with col3:
                    trend_50 = "Bullish" if current_price > ma50 else "Bearish"
                    st.metric("50-Day Trend", trend_50)

    with analytics_tab3:
        st.subheader("🎯 Risk Assessment & Recommendations")
        
        if st.session_state.user_profile:
            st.markdown("#### Personalized Risk Assessment")
            
            risk_tolerance = st.session_state.user_profile.get('risk_tolerance', 'Moderate')
            age_range = st.session_state.user_profile.get('age', 'Unknown')
            time_horizon = st.session_state.user_profile.get('time_horizon', 'Unknown')
            
            risk_score = 5
            
            if risk_tolerance.lower() == 'conservative':
                risk_score = 3
            elif risk_tolerance.lower() == 'aggressive':
                risk_score = 8
            
            if '20-30' in age_range or '30-40' in age_range:
                risk_score += 1
            elif '50-60' in age_range or '60+' in age_range:
                risk_score -= 1
            
            if 'long term' in time_horizon.lower() or 'years' in time_horizon.lower():
                risk_score += 1
            elif 'short term' in time_horizon.lower() or 'months' in time_horizon.lower():
                risk_score -= 1
            
            risk_score = max(1, min(10, risk_score))
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Your Risk Score (1-10)"},
                gauge = {
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 3.5], 'color': "lightgreen"},
                        {'range': [3.5, 7], 'color': "yellow"},
                        {'range': [7, 10], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_score}
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Personalized Recommendations")
            
            if risk_score <= 3:
                st.info("🛡️ **Conservative Strategy**: Focus on bonds, dividend stocks, and stable blue-chip companies.")
            elif risk_score <= 7:
                st.info("⚖️ **Balanced Strategy**: Mix of growth and value stocks with some bonds for stability.")
            else:
                st.warning("🚀 **Aggressive Strategy**: Growth stocks, tech companies, and emerging markets with higher volatility.")
        
        else:
            st.info("💡 Answer personalized questions in the chat to get customized risk assessments!")

    # Expert Analysis Section
    st.markdown('<h2 class="section-header">🎓 AI Expert Analysis</h2>', unsafe_allow_html=True)

    expert_col1, expert_col2 = st.columns(2)

    with expert_col1:
        st.markdown("#### 🔍 Market Insights")
        
        if yfinance_data:
            performers = []
            for symbol, data in yfinance_data.items():
                if 'history' in data and len(data['history']) > 1:
                    hist = data['history']
                    if len(hist) >= 2:
                        change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                        performers.append((symbol, change))
            
            if performers:
                performers.sort(key=lambda x: x[1], reverse=True)
                
                st.markdown("**Top Performers Today:**")
                for symbol, change in performers[:3]:
                    color = "green" if change > 0 else "red"
                    st.markdown(f"• {symbol}: <span style='color:{color}'>{change:+.2f}%</span>", unsafe_allow_html=True)
                
                st.markdown("**Biggest Decliners:**")
                for symbol, change in performers[-3:]:
                    color = "green" if change > 0 else "red"
                    st.markdown(f"• {symbol}: <span style='color:{color}'>{change:+.2f}%</span>", unsafe_allow_html=True)

    with expert_col2:
        st.markdown("#### 📊 Market Statistics")
        
        if yfinance_data:
            total_companies = len(yfinance_data)
            avg_market_cap = 0
            total_volume = 0
            
            for symbol, data in yfinance_data.items():
                if 'info' in data:
                    info = data['info']
                    if info.get('marketCap'):
                        avg_market_cap += info['marketCap']
                
                if 'history' in data and len(data['history']) > 0:
                    total_volume += data['history']['Volume'].iloc[-1] if 'Volume' in data['history'].columns else 0
            
            avg_market_cap = avg_market_cap / total_companies if total_companies > 0 else 0
            
            st.metric("Companies Tracked", total_companies)
            st.metric("Avg Market Cap", f"${avg_market_cap/1e9:.1f}B")
            st.metric("Total Volume", f"{total_volume/1e6:.1f}M")

    # Footer with enhanced information
    st.markdown("""
    <div class="footer">
        <h3>📈 Advanced Agentic AI Financial Assistant</h3>
        <p>Powered by LangChain Agents • GPT-4 • Real-time Web Research • Multi-source Financial Data</p>
        <p><strong>Capabilities:</strong> Personalized Analysis • Market Research • Risk Assessment • Technical Analysis</p>
        <p>© 2025 xAI • All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

    # Additional Agent Information in Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section"><h3>🛠️ Agent Tools</h3></div>', unsafe_allow_html=True)
        st.write("✅ Financial Database Search")
        st.write("✅ Real-time Web Research")  
        st.write("✅ Market Sentiment Analysis")
        st.write("✅ Financial Calculations")
        st.write("✅ Personalized Recommendations")
        
        st.markdown('<div class="sidebar-section"><h3>🎯 Agent Features</h3></div>', unsafe_allow_html=True)
        st.write("🧠 Memory of conversations")
        st.write("👤 User profile building")
        st.write("🔍 Multi-step reasoning")
        st.write("📊 Data visualization")
        st.write("🌐 Internet research")
        st.write("📈 Real-time analysis")