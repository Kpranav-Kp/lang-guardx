"""
lang_guardx/langchain_layer/adapter.py

Layer 2 — LangChain Adapter.
THIS IS THE ONLY FILE IN LAYER 2 THAT IMPORTS LANGCHAIN.

Thin wrapper. No security logic lives here.
If you find yourself writing SQL-checking logic in this file,
it belongs in engine.py instead.
"""

from __future__ import annotations

from langchain_core.language_models import BaseLanguageModel
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from .policy import PolicyVerdict, Verdict
from .engine import SQLPolicyEngine

class ProtectedQueryTool(QuerySQLDataBaseTool):
    policy_engine: SQLPolicyEngine

    def _run(self, query: str) -> str:
        verdict = self.policy_engine.validate(query)
        if verdict.verdict == Verdict.BLOCKED:
            reasons = "; ".join(verdict.violations)
            return f"[LangGuardX] Query blocked by policy. Reasons: {reasons}"
        if verdict.safe_sql is None:
            return "[LangGuardX] Internal error: safe_sql is None on non-blocked verdict"
        return str(self.db.run(verdict.safe_sql))
    
class ProtectedSQLAgent:
    def __init__(
            self,
            llm: BaseLanguageModel,
            db: SQLDatabase,
            engine: SQLPolicyEngine,
    ) -> None:
        self._toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        self._tools = self._toolkit.get_tools()
        
        protected_tools = []
        for tool in self._tools:
            if isinstance(tool, QuerySQLDataBaseTool):
                protected_tools.append(ProtectedQueryTool(db=db, policy_engine=engine))
            else:
                protected_tools.append(tool)
        
        self._agent = create_sql_agent(
            llm=llm,
            toolkit=self._toolkit,
            tools=protected_tools,
            verbose=True,
        )

    def run(self, question : str) -> str:
        result = self._agent.invoke({"input": question})
        return result.get("output", str(result))