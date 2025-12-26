
from server.logger import logger
from langchain_core.messages import AIMessage

def query_chain(chain, user_input: str):
    try:
        logger.debug(f"Running chain for input: {user_input}")

        result = chain.invoke({"query": user_input})

        # ✅ Handle LangChain 1.x output correctly
        if isinstance(result, dict):
            answer = result["result"]
            if isinstance(answer, AIMessage):
                answer = answer.content

            source_docs = result.get("source_documents", [])

        elif isinstance(result, AIMessage):
            answer = result.content
            source_docs = []

        else:
            answer = str(result)
            source_docs = []

        response = {
            "response": answer,
            "sources": [
                doc.metadata.get("source", "")
                for doc in source_docs
            ]
        }

        logger.debug(f"Chain response: {response}")
        return response

    except Exception:
        logger.exception("Error on query chain")
        raise
