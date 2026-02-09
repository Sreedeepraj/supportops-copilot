import asyncio
from typing import Any, Dict, List, Optional

from fastmcp import Client

MCP_URL = "http://127.0.0.1:8765/mcp"


async def _call_tool(tool: str, args: Dict[str, Any]) -> Any:
    async with Client(MCP_URL) as client:
        res = await client.call_tool(tool, args)

        # 1) unwrap result envelope
        payload = None
        if hasattr(res, "data"):
            payload = res.data
        elif hasattr(res, "content"):
            payload = res.content
        else:
            payload = res

        # 2) convert pydantic models -> dict/list
        return _to_plain(payload)


def _to_plain(x: Any) -> Any:
    # Pydantic v2
    if hasattr(x, "model_dump"):
        return x.model_dump()
    # Pydantic v1
    if hasattr(x, "dict"):
        return x.dict()

    if isinstance(x, list):
        return [_to_plain(i) for i in x]
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}

    return x


class MCPTools:
    def memory_search(self, user_id: str, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        return asyncio.run(_call_tool("memory_search", {"user_id": user_id, "query": query, "top_k": top_k}))

    def memory_add(self, user_id: str, session_id: str, question: str, answer: str) -> Dict[str, Any]:
        return asyncio.run(
            _call_tool(
                "memory_add",
                {"user_id": user_id, "session_id": session_id, "question": question, "answer": answer},
            )
        )

    def docs_search(self, query: str, top_k: int = 4, metadata_filter: Optional[dict] = None) -> List[Dict[str, Any]]:
        return asyncio.run(
            _call_tool(
                "docs_search",
                {"query": query, "top_k": top_k, "metadata_filter": metadata_filter},
            )
        )