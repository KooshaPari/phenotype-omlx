"""Unit coverage for gRPC conversion helpers and server lifecycle."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agileplus_mcp.grpc_backlog import AgilePlusBacklogGrpcMixin
from agileplus_mcp.grpc_client import AgilePlusCoreClient, GrpcCallError as ClientGrpcCallError
from agileplus_mcp.grpc_errors import GrpcCallError, GrpcConnectionError
from agileplus_mcp.grpc_serialization import AgilePlusGrpcSerializationMixin


def test_grpc_errors_preserve_code_and_message() -> None:
    connection = GrpcConnectionError("offline")
    call = GrpcCallError(SimpleNamespace(name="INVALID_ARGUMENT"), "bad request")

    assert str(connection) == "offline"
    assert call.code.name == "INVALID_ARGUMENT"
    assert str(call) == "gRPC error namespace(name='INVALID_ARGUMENT'): bad request"


def test_serialization_helpers_convert_repeated_and_binary_fields() -> None:
    feature = SimpleNamespace(
        id=1,
        slug="feature-one",
        friendly_name="Feature One",
        state="planned",
        target_branch="main",
        created_at="created",
        updated_at="updated",
        wp_count=2,
        wp_done=1,
    )
    work_package = SimpleNamespace(
        id=3,
        title="Implement",
        state="doing",
        sequence=1,
        agent_id="agent-1",
        pr_url="https://example.test/pr/1",
        pr_state="open",
        depends_on=(2,),
        file_scope=("src/lib.rs",),
    )
    audit = SimpleNamespace(
        id=4,
        feature_slug="feature-one",
        wp_sequence=1,
        timestamp="now",
        actor="agent-1",
        transition="planned->implementing",
        evidence_refs=("log-1",),
        prev_hash=b"prev",
        hash=b"current",
    )

    assert AgilePlusGrpcSerializationMixin._feature_to_dict(feature)["wp_done"] == 1
    assert AgilePlusGrpcSerializationMixin._wp_to_dict(work_package)["depends_on"] == [2]
    assert AgilePlusGrpcSerializationMixin._wp_to_dict(work_package)["file_scope"] == ["src/lib.rs"]
    assert AgilePlusGrpcSerializationMixin._audit_entry_to_dict(audit)["prev_hash"] == "70726576"
    assert AgilePlusGrpcSerializationMixin._audit_entry_to_dict(audit)["hash"] == "63757272656e74"


def test_backlog_item_conversion_returns_plain_dict() -> None:
    item = SimpleNamespace(
        id=7,
        type="task",
        title="Fix it",
        description="Details",
        priority="high",
        state="queued",
        source="mcp",
        feature_slug="feature-one",
        tags=("bug", "urgent"),
        created_at="created",
        updated_at="updated",
    )

    result = AgilePlusBacklogGrpcMixin._backlog_item_to_dict(item)

    assert result == {
        "id": 7,
        "type": "task",
        "title": "Fix it",
        "description": "Details",
        "priority": "high",
        "state": "queued",
        "source": "mcp",
        "feature_slug": "feature-one",
        "tags": ["bug", "urgent"],
        "created_at": "created",
        "updated_at": "updated",
    }


@pytest.mark.asyncio
async def test_streaming_helper_maps_terminal_event() -> None:
    client = AgilePlusCoreClient()
    client._stub = MagicMock()

    async def events(_request):
        yield SimpleNamespace(
            event_type="updated",
            feature_slug="feature-one",
            wp_sequence=2,
            agent_id="agent-1",
            payload="{}",
            timestamp="now",
        )

    client._stub.StreamAgentEvents = events
    actual = [item async for item in client.stream_agent_events("feature-one")]

    assert actual == [
        {
            "event_type": "updated",
            "feature_slug": "feature-one",
            "wp_sequence": 2,
            "agent_id": "agent-1",
            "payload": "{}",
            "timestamp": "now",
        }
    ]


@pytest.mark.asyncio
async def test_streaming_helper_retries_unavailable(monkeypatch) -> None:
    import grpc

    import agileplus_mcp.grpc_client as grpc_client

    client = AgilePlusCoreClient()
    client._stub = MagicMock()
    client.connect = AsyncMock()
    attempts = 0

    async def events(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise grpc.aio.AioRpcError(
                grpc.StatusCode.UNAVAILABLE,
                initial_metadata=grpc.aio.Metadata(),
                trailing_metadata=grpc.aio.Metadata(),
                details="retry",
            )
        yield SimpleNamespace(
            event_type="updated",
            feature_slug="feature-one",
            wp_sequence=1,
            agent_id="agent-1",
            payload="{}",
            timestamp="now",
        )

    client._stub.StreamAgentEvents = events
    monkeypatch.setattr(grpc_client.asyncio, "sleep", AsyncMock())
    actual = [item async for item in client.stream_agent_events("feature-one")]

    assert attempts == 2
    client.connect.assert_awaited_once()
    assert actual[0]["feature_slug"] == "feature-one"


@pytest.mark.asyncio
async def test_streaming_helper_maps_terminal_rpc_error() -> None:
    import grpc

    client = AgilePlusCoreClient()

    async def events(_request):
        raise grpc.aio.AioRpcError(
            grpc.StatusCode.INVALID_ARGUMENT,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="bad request",
        )
        yield  # pragma: no cover

    client._stub = MagicMock()
    client._stub.StreamAgentEvents = events

    with pytest.raises(ClientGrpcCallError, match="bad request"):
        _ = [item async for item in client.stream_agent_events("feature-one")]


@pytest.mark.asyncio
async def test_server_startup_registers_client_and_shutdown_closes(monkeypatch) -> None:
    import agileplus_mcp.server as server

    client = MagicMock()
    client.connect = AsyncMock()
    client.close = AsyncMock()

    monkeypatch.setattr(server, "_client", None)
    monkeypatch.setattr(server, "_sampling", None)
    monkeypatch.setattr(server, "AgilePlusCoreClient", MagicMock(return_value=client))
    monkeypatch.setattr(server.features_module, "register_tools", MagicMock())
    monkeypatch.setattr(server.governance_module, "register_tools", MagicMock())
    monkeypatch.setattr(server.status_module, "register_tools", MagicMock())

    await server.startup("localhost:50051")
    await server.shutdown()

    client.connect.assert_awaited_once()
    client.close.assert_awaited_once()
    assert server._client is None
