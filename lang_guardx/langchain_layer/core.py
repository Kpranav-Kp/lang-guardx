from __future__ import annotations
from typing import Any, Optional, Dict, List, Union, TYPE_CHECKING
from lang_guardx.langchain_layer.sql_policy import (
    SQLPolicy,
    SQLPolicyEngine,
    PolicyVerdict,
)
from lang_guardx.langchain_layer.chain import (
    ProtectedChain,
    ChainTrace,
    protect_chain,
)
from lang_guardx.langchain_layer.agent import (
    ProtectedAgent,
    AgentTrace,
    protect_agent,
)

if TYPE_CHECKING:
    from lang_guardx.detection.core import Detector


# ---------------------------------------------------------------------------
# Type alias for the trace returned by either chain or agent
# ---------------------------------------------------------------------------
ExecutionTrace = Union[ChainTrace, AgentTrace]


# ---------------------------------------------------------------------------
# Layer 2 coordinator
# ---------------------------------------------------------------------------

class LangGuardXLayer2:
    """
    Single coordinator for all Layer 2 security decisions.

    Internally wraps either a ProtectedChain or ProtectedAgent
    depending on how it was created. Exposes a single .run() interface
    regardless of which mode is active.

    Create via class methods:
        LangGuardXLayer2.for_chain(...)   — for SQLDatabaseChain
        LangGuardXLayer2.for_agent(...)   — for AgentExecutor
    """

    def __init__(
        self,
        protected: Union[ProtectedChain, ProtectedAgent],
        policy: SQLPolicy,
        mode: str,           # "chain" | "agent"
    ) -> None:
        self._protected = protected
        self._policy = policy
        self._mode = mode

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def for_chain(
        cls,
        chain: Any,
        current_user_id: Optional[int] = None,
        system_prompt: str = "",
        policy: Optional[SQLPolicy] = None,
        detector: Optional["Detector"] = None,
        # Convenience kwargs — used when policy is not pre-built
        permitted_tables: Optional[List[str]] = None,
        restricted_columns: Optional[Dict[str, List[str]]] = None,
        require_user_scope: bool = False,
        scoped_tables: Optional[List[str]] = None,
        max_rows: int = 50,
    ) -> "LangGuardXLayer2":
        """
        Create a Layer 2 instance wrapping a LangChain SQLDatabaseChain.

        Args:
            chain:              The SQLDatabaseChain to protect.
            current_user_id:    Authenticated user ID for SQL user-scope
                                rewriting. Pass None to disable.
            system_prompt:      Application system prompt. Used for
                                structured prompt formatting (PromptGuard
                                Layer 2) and passed to Layer 3.
            policy:             Pre-built SQLPolicy. If provided, the
                                convenience kwargs below are ignored.
            detector:           Pre-instantiated Layer 1 Detector.
                                If None, a new Detector is created
                                (loads DistilBERT from disk).
            permitted_tables:   Tables the LLM may query.
            restricted_columns: {table: [col, col]} — never returned.
            require_user_scope: Inject WHERE user_id = current_user.
            scoped_tables:      Which tables get user-scope injected.
            max_rows:           Hard cap on LIMIT clause.

        Returns:
            LangGuardXLayer2 in chain mode.
        """
        resolved_policy = policy or cls._build_policy(
            permitted_tables=permitted_tables,
            restricted_columns=restricted_columns,
            require_user_scope=require_user_scope,
            scoped_tables=scoped_tables,
            max_rows=max_rows,
        )
        protected = protect_chain(
            chain=chain,
            policy=resolved_policy,
            current_user_id=current_user_id,
            system_prompt=system_prompt,
            detector=detector,
        )
        return cls(protected=protected, policy=resolved_policy, mode="chain")

    @classmethod
    def for_agent(
        cls,
        agent_executor: Any,
        current_user_id: Optional[int] = None,
        system_prompt: str = "",
        policy: Optional[SQLPolicy] = None,
        detector: Optional["Detector"] = None,
        raise_on_dangerous_tools: bool = False,
        # Convenience kwargs
        permitted_tables: Optional[List[str]] = None,
        restricted_columns: Optional[Dict[str, List[str]]] = None,
        require_user_scope: bool = False,
        scoped_tables: Optional[List[str]] = None,
        max_rows: int = 50,
    ) -> "LangGuardXLayer2":
        """
        Create a Layer 2 instance wrapping a LangChain AgentExecutor.

        Args:
            agent_executor:           The AgentExecutor to protect.
            current_user_id:          Authenticated user ID.
            system_prompt:            Application system prompt.
            policy:                   Pre-built SQLPolicy.
            detector:                 Pre-instantiated Layer 1 Detector.
            raise_on_dangerous_tools: If True, raise ValueError when
                                      dangerous tools are detected at
                                      startup instead of warning.
            permitted_tables:         Tables the LLM may query.
            restricted_columns:       {table: [col, col]} — never returned.
            require_user_scope:       Inject WHERE user_id = current_user.
            scoped_tables:            Which tables get user-scope injected.
            max_rows:                 Hard cap on LIMIT clause.

        Returns:
            LangGuardXLayer2 in agent mode.
        """
        resolved_policy = policy or cls._build_policy(
            permitted_tables=permitted_tables,
            restricted_columns=restricted_columns,
            require_user_scope=require_user_scope,
            scoped_tables=scoped_tables,
            max_rows=max_rows,
        )
        protected = protect_agent(
            agent_executor=agent_executor,
            policy=resolved_policy,
            current_user_id=current_user_id,
            system_prompt=system_prompt,
            detector=detector,
            raise_on_dangerous_tools=raise_on_dangerous_tools,
        )
        return cls(protected=protected, policy=resolved_policy, mode="agent")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> tuple[str, ExecutionTrace]:
        """
        Run the protected chain or agent on user_input.

        Returns:
            (response_string, ExecutionTrace)

            ExecutionTrace is either ChainTrace or AgentTrace depending
            on the mode. Both are accepted by Layer 3's Validator.check().

        Example:
            response, trace = layer2.run("show me laptops under $1000")

            # Pass trace to Layer 3
            verdict = validator.check(response, user_input, trace)
        """
        return self._protected.run(user_input)

    def validate_sql(self, sql: str) -> PolicyVerdict:
        """
        Validate a raw SQL string against the active policy.

        Useful for testing policy configuration without running
        the full chain/agent. Also used by the evaluation harness
        in test_comparison_study.py.

        Example:
            verdict = layer2.validate_sql("SELECT * FROM users")
            print(verdict.blocked)       # True
            print(verdict.violations)    # ["wildcard_select: ..."]
            print(verdict.safe_sql)      # None (blocked)
        """
        engine = self._get_policy_engine()
        return engine.validate(sql)

    @property
    def mode(self) -> str:
        """Returns "chain" or "agent"."""
        return self._mode

    @property
    def policy(self) -> SQLPolicy:
        """Returns the active SQLPolicy configuration."""
        return self._policy

    def policy_summary(self) -> str:
        """
        Human-readable summary of the active policy.
        Useful for logging and debugging during development.

        Example output:
            Layer 2 Policy Summary
            ─────────────────────────────────────
            Mode:               chain
            Permitted ops:      ['SELECT']
            Permitted tables:   ['products', 'reviews']
            Restricted cols:    {'users': ['password', 'email']}
            User scope:         enabled (column: user_id)
            Scoped tables:      ['orders']
            Max rows:           50
        """
        p = self._policy
        lines = [
            "Layer 2 Policy Summary",
            "─" * 37,
            f"Mode:               {self._mode}",
            f"Permitted ops:      {p.permitted_operations}",
            f"Permitted tables:   {p.permitted_tables or 'all tables'}",
            f"Restricted cols:    {p.restricted_columns or 'none'}",
            f"User scope:         {'enabled (column: ' + p.user_scope_column + ')' if p.require_user_scope else 'disabled'}",
            f"Scoped tables:      {p.scoped_tables or 'all (if scope enabled)'}",
            f"Max rows:           {p.max_rows}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_policy(
        permitted_tables: Optional[List[str]],
        restricted_columns: Optional[Dict[str, List[str]]],
        require_user_scope: bool,
        scoped_tables: Optional[List[str]],
        max_rows: int,
    ) -> SQLPolicy:
        """Build a SQLPolicy from convenience kwargs."""
        return SQLPolicy(
            permitted_operations=["SELECT"],
            permitted_tables=permitted_tables or [],
            restricted_columns=restricted_columns or {},
            require_user_scope=require_user_scope,
            scoped_tables=scoped_tables or [],
            max_rows=max_rows,
        )

    def _get_policy_engine(self) -> SQLPolicyEngine:
        """Extract the SQLPolicyEngine from the wrapped protected object."""
        return self._protected._policy_engine