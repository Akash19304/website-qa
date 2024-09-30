website rag streaming chatbot with langchain implemented separately with a `Streamlit` App & `FastApi`.

create virtual environment 
``
pip install -r requirements.txt
``

Enter the `huggingface API token` and `Groq API key` in the `.env` file.

FastApi endpoints: 
- `/ask` - input a question and get answer.

- `/docs` - for FastApi swagger ui.
