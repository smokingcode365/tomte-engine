"""One native model repair, retaining independently valid proposal items."""
from contextvars import ContextVar
from types import SimpleNamespace

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior

from tomte.engine.core.proposals import ProposalResponse


_run = ContextVar("proposal_recovery_run", default=None)


def recoverable(schema):
    return isinstance(schema, type) and issubclass(schema, ProposalResponse)


def install_recovery(agent, schema):
    if not recoverable(schema):
        return agent

    @agent.output_validator
    def validate(ctx: RunContext, response):
        state = _run.get()
        if state is None or not isinstance(response, ProposalResponse):
            return response
        # Keep the public RunContext history so a malformed second output is
        # still represented in retry diagnostics when we retain the first result.
        state["messages"] = ctx.messages
        if state["first"] is None:
            state["first"] = response
            if response._rejected and ctx.retry < 1:
                response._repair_attempted = True
                raise ModelRetry(
                    "Some proposal items failed structural validation at these paths: "
                    + repr([{"path": e["loc"], "code": e["type"]}
                            for e in response._validation_errors])
                    + ". Repair only these items. Return the complete response in the same "
                    "order and positions; preserve accepted items, target/correlation IDs "
                    "and operation kinds. Use only the original user evidence. Do not "
                    "invent missing facts or IDs. If unsupported, leave the item unchanged "
                    "for rejection; do not replace it with another proposal.")
            response._repair_attempted = ctx.retry > 0
            return response
        return state["first"].merge_repair(response)

    return agent


def run_with_recovery(agent, schema, user_text, **kwargs):
    if not recoverable(schema):
        return agent.run_sync(user_text, **kwargs)
    state = {"first": None, "messages": []}
    token = _run.set(state)
    try:
        try:
            result = agent.run_sync(user_text, **kwargs)
        except UnexpectedModelBehavior:
            # The bounded repair may itself be malformed. Retain only what was
            # independently valid before repair. Transport/storage/control errors
            # are deliberately not caught here.
            if state["first"] is None:
                raise
            result = SimpleNamespace(output=state["first"], all_messages=lambda: state["messages"])
        if isinstance(result.output, ProposalResponse) and result.output._rejected:
            try:
                from tomte.engine import tomtelog
                tomtelog.warn(f"{schema.__name__}: skipped {len(result.output._rejected)} invalid proposal items after bounded repair")
            except Exception:
                pass
        return result
    finally:
        _run.reset(token)
