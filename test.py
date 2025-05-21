from flask import Flask, request, render_template_string
import os
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Setup Flask
app = Flask(__name__)

# تحميل مفاتيح API من ملف .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
OER_API_KEY = os.getenv("EXCHANGERATES_API_KEY")

# تحميل البيانات المالية من ملف CSV
csv_path = 'financial_risk_analysis_large.csv'
financial_data = pd.read_csv(csv_path)

# جلب بيانات Alpha Vantage

def get_stock_data(symbols=["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "META", "JPM", "WMT", "PG"]):
    all_data = {}
    for symbol in symbols:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_API_KEY}&outputsize=compact"
        try:
            response = requests.get(url)
            data = response.json()
            if 'Time Series (Daily)' in data:
                df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index').astype(float)
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                all_data[symbol] = df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    return all_data

alpha_data = get_stock_data()

# جلب بيانات FMP

def get_company_profile(symbols=["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "META", "JPM", "WMT", "PG"]):
    all_data = {}
    for symbol in symbols:
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={FMP_API_KEY}"
        try:
            response = requests.get(url)
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                all_data[symbol] = data[0]
        except Exception as e:
            print(f"Error fetching profile data for {symbol}: {e}")
    return all_data

fmp_data = get_company_profile()

# جلب بيانات أسعار الصرف

def get_exchange_rate(base="USD", symbols="EUR"):
    url = f"https://openexchangerates.org/api/latest.json?app_id={OER_API_KEY}&base={base}&symbols={symbols}"
    response = requests.get(url)
    try:
        return response.json()
    except:
        return {}

exchange_rate_data = get_exchange_rate()

# جلب بيانات البنك الدولي وEurostat

def get_worldbank_inflation_data(country="USA", indicator="FP.CPI.TOTL.ZG", start_year=2000, end_year=2024):
    url = f"http://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&date={start_year}:{end_year}"
    try:
        response = requests.get(url)
        data = response.json()
        if data and len(data) > 1 and isinstance(data[1], list):
            df = pd.DataFrame(data[1])
            df['date'] = pd.to_datetime(df['date'])
            return df[['date', 'value', 'country', 'indicator']].dropna()
    except:
        return pd.DataFrame()

def get_eurostat_inflation_data(dataset="prc_hicp_midx", geo="EA", start_period="2010-01"):
    url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}?format=JSON&geo={geo}&unit=CP00&startPeriod={start_period}"
    try:
        response = requests.get(url)
        data = response.json()
        if 'value' in data and 'dimension' in data:
            dates = list(data['dimension']['time']['category']['index'].keys())
            values = [float(data['value'].get(str(i), None)) for i in range(len(dates)) if str(i) in data['value']]
            return pd.DataFrame({
                'date': pd.to_datetime(dates),
                'value': values,
                'indicator': 'HICP',
                'region': geo
            }).dropna()
    except:
        return pd.DataFrame()

combined_data = {
    "csv_data": financial_data.to_dict(orient="records"),
    "stock_data_alpha": alpha_data,
    "company_profile_fmp": fmp_data,
    "exchange_rate": exchange_rate_data,
    "worldbank_inflation": get_worldbank_inflation_data().to_dict(orient="records"),
    "eurostat_inflation": get_eurostat_inflation_data().to_dict(orient="records")
}

def convert_data_to_documents(combined_data):
    documents = []
    for row in combined_data["csv_data"][:100]:
        content = "\n".join([f"{k}: {v}" for k, v in row.items() if v is not None])
        documents.append(Document(page_content=content, metadata={"source": "csv"}))
    documents.append(Document(page_content=str(combined_data["stock_data_alpha"]), metadata={"source": "alpha_vantage"}))
    documents.append(Document(page_content=str(combined_data["company_profile_fmp"]), metadata={"source": "fmp"}))
    documents.append(Document(page_content=str(combined_data["exchange_rate"]), metadata={"source": "exchange_rates"}))
    for row in combined_data["worldbank_inflation"]:
        content = f"Inflation Rate in {row.get('country', 'Unknown')}: {row.get('value', 'N/A')}% in {row.get('date', 'N/A').year}. "\
                  f"Indicator: {row.get('indicator', 'N/A')}. High inflation can lead to reduced consumer purchasing power, increased borrowing costs, currency depreciation, and economic slowdown."
        documents.append(Document(page_content=content, metadata={"source": "worldbank"}))
    for row in combined_data["eurostat_inflation"]:
        content = f"Inflation Rate (HICP) in {row.get('region', 'Unknown')}: {row.get('value', 'N/A')}% in {row.get('date', 'N/A').strftime('%Y-%m')}. "\
                  f"High inflation may reduce consumer spending, tighten monetary policy, cause currency fluctuations, and risk economic instability."
        documents.append(Document(page_content=content, metadata={"source": "eurostat"}))
    return documents

documents = convert_data_to_documents(combined_data)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(split_docs, embeddings)

prompt_template = """
Use the following context to answer the question. Answer **only** using the information provided in the context.
If the answer is not present in the context, clearly say that there is not enough information to answer.
Respond in the same language as the question — Arabic or English.

استخدم السياق التالي للإجابة على السؤال. أجب فقط باستخدام المعلومات الموجودة في السياق.
إذا لم تكن المعلومات كافية للإجابة، فاذكر ذلك بوضوح.
أجب بنفس لغة السؤال - العربية أو الإنجليزية.

Context | السياق:
{context}

Question | السؤال:
{question}

Answer | الإجابة:
"""


prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        question = request.form.get("question")
        if question:
            response = rag_chain.invoke({"query": question})
            result = response["result"]
    return render_template_string('''
    <form method="post">
        <input type="text" name="question" placeholder="اكتب سؤالك هنا" style="width:400px"/>
        <input type="submit" value="إرسال"/>
    </form>
    <div style="margin-top:20px">
        <b>الإجابة:</b>
        <p>{{ result }}</p>
    </div>
    ''', result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
