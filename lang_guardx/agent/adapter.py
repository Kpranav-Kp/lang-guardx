from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent

from .policy import Verdict
from .engine import SQLPolicyEngine


class _PolicyEnforcedDatabase(SQLDatabase):
    """
    Composition wrapper around SQLDatabase.
    Intercepts every db.run() call and passes it through SQLPolicyEngine
    before any SQL reaches the database. Does NOT inherit SQLDatabase —
    uses __getattr__ delegation instead to avoid shallow-copy fragility.
    """

    def __init__(self, db: SQLDatabase, engine: SQLPolicyEngine):
        object.__setattr__(self, "_db", db)
        object.__setattr__(self, "_guard", engine)
        object.__setattr__(self, "_last_blocked", False)

    def run(self, command: str, *args, **kwargs) -> str:
        self._last_blocked = False
        verdict = self._guard.validate(command)
        if verdict.verdict == Verdict.BLOCKED:
            self._last_blocked = True
            return f"[LangGuardX BLOCKED] {'; '.join(verdict.violations)}"
        return str(self._db.run(str(verdict.safe_sql), *args, **kwargs))

    def __getattr__(self, name: str):
        # All attributes not found on this wrapper delegate to the real db.
        # This covers: dialect, get_table_info, get_usable_table_names, etc.
        return getattr(self._db, name)

    @property
    def last_blocked(self) -> bool:
        return self._last_blocked

_SYSTEM_PROMPT = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results and return the answer.

Rules:
- Always look at the available tables first.
- Never query all columns — only ask for relevant columns.
- Limit results to at most {top_k} rows unless the user specifies otherwise.
- If a query is blocked by policy, tell the user clearly. Do NOT retry.
"""


class ProtectedSQLAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        db: SQLDatabase,
        engine: SQLPolicyEngine,
        top_k: int = 10,
    ) -> None:
        self._protected_db = _PolicyEnforcedDatabase(db, engine)

        toolkit = SQLDatabaseToolkit(db=self._protected_db, llm=llm)
        tools = toolkit.get_tools()

        system_prompt = _SYSTEM_PROMPT.format(
            dialect=db.dialect,
            top_k=top_k,
        )

        self._agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )

    def run(self, question: str) -> str:
        result = self._agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        messages = result.get("messages", [])
        return messages[-1].content if messages else str(result)

    @property
    def last_blocked(self) -> bool:
        return self._protected_db.last_blocked