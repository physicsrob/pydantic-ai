import uuid
from dataclasses import dataclass
from typing import Any, cast

import anyio
import httpx
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart as PydanticAITextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import Usage

from .conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from fasta2a.broker import InMemoryBroker, StreamEvent
    from fasta2a.client import A2AClient
    from fasta2a.schema import DataPart, FilePart, Message, TaskSendParams, TextPart
    from fasta2a.storage import InMemoryStorage

    from pydantic_ai._a2a import agent_to_a2a


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='fasta2a not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


def return_string(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    assert info.output_tools is not None
    args_json = '{"response": ["foo", "bar"]}'
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])


model = FunctionModel(return_string)


async def wait_for_task(client: A2AClient, task_id: str, timeout_loops: int = 10) -> Any:
    """Wait for a task to complete or fail, using polling pattern."""
    for _ in range(timeout_loops):
        task_response = await client.get_task(task_id)
        if task_response and 'result' in task_response:
            task = task_response['result']
            if task['status']['state'] in ('completed', 'failed'):
                return task
        await anyio.sleep(0.1)

    # If we get here, task didn't complete in time
    pytest.fail(f'Task {task_id} did not complete within {timeout_loops * 0.1} seconds')


# Define a test Pydantic model
class UserProfile(BaseModel):
    name: str
    age: int
    email: str


def return_pydantic_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    assert info.output_tools is not None
    args_json = '{"name": "John Doe", "age": 30, "email": "john@example.com"}'
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])


pydantic_model = FunctionModel(return_pydantic_model)


async def test_a2a_pydantic_model_output():
    """Test that Pydantic model outputs have correct metadata including JSON schema."""
    agent = Agent(model=pydantic_model, output_type=UserProfile)
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Get user profile', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'

            task_id = result['id']

            # Wait for completion
            result = await wait_for_task(a2a_client, task_id)
            assert result['status']['state'] == 'completed'

            # Check artifacts
            assert 'artifacts' in result
            assert len(result['artifacts']) == 1
            artifact = result['artifacts'][0]

            # Verify the data
            assert artifact['parts'][0]['kind'] == 'data'
            assert artifact['parts'][0]['data'] == {
                'result': {'name': 'John Doe', 'age': 30, 'email': 'john@example.com'}
            }

            metadata = artifact['parts'][0].get('metadata')
            assert metadata is not None

            assert metadata['json_schema'] == snapshot(
                {
                    'properties': {
                        'name': {'title': 'Name', 'type': 'string'},
                        'age': {'title': 'Age', 'type': 'integer'},
                        'email': {'title': 'Email', 'type': 'string'},
                    },
                    'required': ['name', 'age', 'email'],
                    'title': 'UserProfile',
                    'type': 'object',
                }
            )

            assert result.get('history') == snapshot(
                [
                    {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': 'Get user profile'}],
                        'kind': 'message',
                        'message_id': IsStr(),
                        'context_id': IsStr(),
                        'task_id': IsStr(),
                    }
                ]
            )


async def test_a2a_runtime_error_without_lifespan():
    agent = Agent(model=model, output_type=tuple[str, str])
    app = agent.to_a2a()

    transport = httpx.ASGITransport(app)
    async with httpx.AsyncClient(transport=transport) as http_client:
        a2a_client = A2AClient(http_client=http_client)

        message = Message(
            role='user',
            parts=[TextPart(text='Hello, world!', kind='text')],
            kind='message',
            message_id=str(uuid.uuid4()),
        )

        with pytest.raises(RuntimeError, match='TaskManager was not properly initialized.'):
            await a2a_client.send_message(message=message)


async def test_a2a_simple():
    agent = Agent(model=model, output_type=tuple[str, str])
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello, world!', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                }
            )

            task_id = result['id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)
            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [
                                    {
                                        'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                        'kind': 'data',
                                        'data': {'result': ['foo', 'bar']},
                                    }
                                ],
                            }
                        ],
                    },
                }
            )


async def test_a2a_file_message_with_file():
    agent = Agent(model=model, output_type=tuple[str, str])
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[
                    FilePart(
                        kind='file',
                        file={'uri': 'https://example.com/file.txt', 'mime_type': 'text/plain'},
                    )
                ],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [
                                {
                                    'kind': 'file',
                                    'file': {'mime_type': 'text/plain', 'uri': 'https://example.com/file.txt'},
                                }
                            ],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                }
            )

            task_id = result['id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)
            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [
                                    {
                                        'kind': 'file',
                                        'file': {'mime_type': 'text/plain', 'uri': 'https://example.com/file.txt'},
                                    }
                                ],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [
                                    {
                                        'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                        'kind': 'data',
                                        'data': {'result': ['foo', 'bar']},
                                    }
                                ],
                            }
                        ],
                    },
                }
            )


async def test_a2a_file_message_with_file_content():
    agent = Agent(model=model, output_type=tuple[str, str])
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[
                    FilePart(file={'bytes': 'foo', 'mime_type': 'text/plain'}, kind='file'),
                ],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'kind': 'file', 'file': {'bytes': 'foo', 'mime_type': 'text/plain'}}],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                }
            )

            task_id = result['id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)
            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'file', 'file': {'bytes': 'foo', 'mime_type': 'text/plain'}}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [
                                    {
                                        'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                        'kind': 'data',
                                        'data': {'result': ['foo', 'bar']},
                                    }
                                ],
                            }
                        ],
                    },
                }
            )


async def test_a2a_file_message_with_data():
    agent = Agent(model=model, output_type=tuple[str, str])
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[DataPart(kind='data', data={'foo': 'bar'})],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'kind': 'data', 'data': {'foo': 'bar'}}],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                }
            )

            task_id = result['id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'failed':
                    break
                await anyio.sleep(0.1)
            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'failed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'data', 'data': {'foo': 'bar'}}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                    },
                }
            )


async def test_a2a_error_handling():
    """Test that errors during task execution properly update task state."""

    def raise_error(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('Test error during agent execution')

    error_model = FunctionModel(raise_error)
    agent = Agent(model=error_model, output_type=str)
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello, world!', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'

            task_id = result['id']

            # Wait for task to fail
            task_result = await wait_for_task(a2a_client, task_id)
            assert task_result['status']['state'] == 'failed'


async def test_a2a_multiple_tasks_same_context():
    """Test that multiple tasks can share the same context_id with accumulated history."""

    messages_received: list[list[ModelMessage]] = []

    def track_messages(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Store a copy of the messages received by the model
        messages_received.append(messages.copy())
        # Return the standard response
        assert info.output_tools is not None
        args_json = '{"response": ["foo", "bar"]}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    tracking_model = FunctionModel(track_messages)
    agent = Agent(model=tracking_model, output_type=tuple[str, str])
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            # First message - should create a new context
            message1 = Message(
                role='user',
                parts=[TextPart(text='First message', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response1 = await a2a_client.send_message(message=message1)
            assert 'error' not in response1
            assert 'result' in response1
            result1 = response1['result']
            assert result1['kind'] == 'task'

            task1_id = result1['id']
            context_id = result1['context_id']

            # Wait for first task to complete
            task1_result = await wait_for_task(a2a_client, task1_id)
            assert task1_result['status']['state'] == 'completed'

            # Verify the model received at least one message
            assert len(messages_received) == 1
            first_run_history = messages_received[0]
            assert first_run_history == snapshot(
                [ModelRequest(parts=[UserPromptPart(content='First message', timestamp=IsDatetime())])]
            )

            # Second message - reuse the same context_id
            message2 = Message(
                role='user',
                parts=[TextPart(text='Second message', kind='text')],
                kind='message',
                context_id=context_id,
                message_id=str(uuid.uuid4()),
            )
            response2 = await a2a_client.send_message(message=message2)
            assert 'error' not in response2
            assert 'result' in response2
            result2 = response2['result']
            assert result2['kind'] == 'task'

            task2_id = result2['id']
            # Verify we got a new task ID but same context ID
            assert task2_id != task1_id
            assert result2['context_id'] == context_id

            # Wait for second task to complete
            while task2 := await a2a_client.get_task(task2_id):  # pragma: no branch
                if 'result' in task2 and task2['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)

            # Verify the model received the full history on the second call
            assert len(messages_received) == 2
            second_run_history = messages_received[1]
            assert second_run_history[0] == first_run_history[0]

            assert second_run_history == snapshot(
                [
                    ModelRequest(parts=[UserPromptPart(content='First message', timestamp=IsDatetime())]),
                    ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name='final_result', args='{"response": ["foo", "bar"]}', tool_call_id=IsStr()
                            )
                        ],
                        usage=Usage(requests=1, request_tokens=52, response_tokens=7, total_tokens=59),
                        model_name='function:track_messages:',
                        timestamp=IsDatetime(),
                    ),
                    ModelRequest(
                        parts=[
                            ToolReturnPart(
                                tool_name='final_result',
                                content='Final result processed.',
                                tool_call_id=IsStr(),
                                timestamp=IsDatetime(),
                            )
                        ]
                    ),
                    ModelRequest(parts=[UserPromptPart(content='Second message', timestamp=IsDatetime())]),
                ]
            )


async def test_a2a_thinking_response():
    """Test that ModelResponse messages with ThinkingPart are properly handled."""

    def return_thinking_response(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        # Create a response with thinking part and text part
        return ModelResponse(
            parts=[
                ThinkingPart(content='Let me think about this...', id='thinking_1'),
                PydanticAITextPart(content="Here's my response"),
            ]
        )

    thinking_model = FunctionModel(return_thinking_response)
    agent = Agent(model=thinking_model, output_type=str)
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello, world!', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'

            task_id = result['id']

            # Wait for completion
            task_result = await wait_for_task(a2a_client, task_id)
            assert task_result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        },
                        {
                            'role': 'agent',
                            'parts': [
                                {
                                    'metadata': {'type': 'thinking', 'thinking_id': 'thinking_1', 'signature': None},
                                    'kind': 'text',
                                    'text': 'Let me think about this...',
                                },
                                {'kind': 'text', 'text': "Here's my response"},
                            ],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        },
                    ],
                    'artifacts': [
                        {
                            'artifact_id': IsStr(),
                            'name': 'result',
                            'parts': [{'kind': 'text', 'text': "Here's my response"}],
                        }
                    ],
                }
            )


async def test_a2a_multiple_messages():
    agent = Agent(model=model, output_type=tuple[str, str])
    storage = InMemoryStorage()
    app = agent.to_a2a(storage=storage)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello, world!', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert response == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': IsStr(),
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                    },
                }
            )

            # NOTE: We include the agent history before we start working on the task.
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            task_id = result['id']
            task = storage.tasks[task_id]
            assert 'history' in task
            task['history'].append(
                Message(
                    role='agent',
                    parts=[TextPart(text='Whats up?', kind='text')],
                    kind='message',
                    message_id=str(uuid.uuid4()),
                )
            )

            response = await a2a_client.get_task(task_id)
            assert response == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            },
                            {
                                'role': 'agent',
                                'parts': [{'kind': 'text', 'text': 'Whats up?'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                            },
                        ],
                    },
                }
            )

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)

            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            },
                            {
                                'role': 'agent',
                                'parts': [{'kind': 'text', 'text': 'Whats up?'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                            },
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [
                                    {
                                        'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                        'kind': 'data',
                                        'data': {'result': ['foo', 'bar']},
                                    }
                                ],
                            }
                        ],
                    },
                }
            )


async def test_a2a_multiple_send_task_messages():
    agent = Agent(model=model, output_type=tuple[str, str])
    storage = InMemoryStorage()
    app = agent.to_a2a(storage=storage)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello, world!', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert response == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': IsStr(),
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                    },
                }
            )
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            task_id = result['id']
            context_id = result['context_id']

            await anyio.sleep(0.1)
            response = await a2a_client.get_task(task_id)
            assert response.get('result') == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                    'artifacts': [
                        {
                            'artifact_id': IsStr(),
                            'name': 'result',
                            'parts': [
                                {
                                    'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                    'kind': 'data',
                                    'data': {'result': ['foo', 'bar']},
                                }
                            ],
                        }
                    ],
                }
            )

            message = Message(
                role='user',
                parts=[TextPart(text='Did you get my first message?', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
                context_id=context_id,
            )
            response = await a2a_client.send_message(message=message)
            assert response == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': IsStr(),
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'text', 'text': 'Did you get my first message?'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                    },
                }
            )

            await anyio.sleep(0.1)
            response = await a2a_client.get_task(task_id)
            assert response.get('result') == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                    'artifacts': [
                        {
                            'artifact_id': IsStr(),
                            'name': 'result',
                            'parts': [
                                {
                                    'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                    'kind': 'data',
                                    'data': {'result': ['foo', 'bar']},
                                }
                            ],
                        }
                    ],
                }
            )


async def test_streaming_emits_incremental_messages(mocker: Any) -> None:
    """Verify that enable_streaming=True produces incremental messages during agent execution."""
    from fasta2a.broker import InMemoryBroker

    # Create a model that produces multiple text parts to simulate streaming
    def return_multiple_text_parts(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                PydanticAITextPart(content='First part of response'),
                PydanticAITextPart(content='Second part of response'),
                PydanticAITextPart(content='Final part'),
            ]
        )

    streaming_model = FunctionModel(return_multiple_text_parts)

    # Create agent with streaming enabled
    agent = Agent(model=streaming_model, output_type=str)
    storage = InMemoryStorage()
    broker = InMemoryBroker()

    # Spy on the broker's send_stream_event method to capture calls
    mock_send: Any = mocker.spy(broker, 'send_stream_event')

    app = agent.to_a2a(enable_streaming=True, storage=storage, broker=broker)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello, world!', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response

            result = response['result']
            assert result['kind'] == 'task'
            task_id = result['id']

            # Wait for task completion
            await wait_for_task(a2a_client, task_id)

            # Verify streaming events were captured
            assert mock_send.call_count > 0

            # Extract events from mock calls
            captured_events: list[StreamEvent] = [call.args[1] for call in mock_send.call_args_list]

            # Check that we got different types of events
            event_kinds = [event.get('kind') for event in captured_events if event.get('kind')]

            # Look for agent messages
            message_events: list[Message] = []
            for event in captured_events:
                if event.get('kind') == 'message' and event.get('role') == 'agent':
                    message_events.append(cast(Message, event))

            # Should have status updates at minimum
            assert 'status-update' in event_kinds

            # Verify we got at least one agent message during streaming
            assert len(message_events) > 0, f'Expected agent messages during streaming, got events: {event_kinds}'

            # Verify the agent message contains the expected content
            agent_message = message_events[0]
            assert agent_message['role'] == 'agent'
            assert agent_message['kind'] == 'message'
            assert 'message_id' in agent_message
            assert 'parts' in agent_message
            parts = agent_message['parts']
            assert len(parts) == 3  # Should have 3 text parts
            first_part = parts[0]
            assert first_part.get('kind') == 'text'
            assert first_part.get('text') == 'First part of response'


async def test_streaming_disabled_sends_only_final_results(mocker: Any) -> None:
    """Verify enable_streaming=False sends only status updates and final results, no incremental messages."""
    from fasta2a.broker import InMemoryBroker

    # Create a model that produces multiple text parts - same as streaming test for comparison
    def return_multiple_text_parts(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                PydanticAITextPart(content='First part of response'),
                PydanticAITextPart(content='Second part of response'),
                PydanticAITextPart(content='Final part'),
            ]
        )

    streaming_model = FunctionModel(return_multiple_text_parts)

    # Create agent with streaming DISABLED (default behavior)
    agent = Agent(model=streaming_model, output_type=str)
    storage = InMemoryStorage()
    broker = InMemoryBroker()

    # Spy on the broker's send_stream_event method to capture calls
    mock_send: Any = mocker.spy(broker, 'send_stream_event')

    # Note: enable_streaming defaults to False, but being explicit for clarity
    app = agent.to_a2a(enable_streaming=False, storage=storage, broker=broker)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello, world!', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response

            result = response['result']
            assert result['kind'] == 'task'
            task_id = result['id']

            # Wait for task completion
            await wait_for_task(a2a_client, task_id)

            # Verify streaming events were captured
            assert mock_send.call_count > 0

            # Extract events from mock calls
            captured_events: list[StreamEvent] = [call.args[1] for call in mock_send.call_args_list]

            # Analyze event types
            event_kinds = [event.get('kind') for event in captured_events if event.get('kind')]

            # Look for any agent messages (should be NONE when streaming disabled)
            agent_messages: list[Message] = []
            for event in captured_events:
                if event.get('kind') == 'message' and event.get('role') == 'agent':
                    agent_messages.append(cast(Message, event))

            # Verify expected behavior: only status updates, NO incremental agent messages
            assert 'status-update' in event_kinds, 'Should have status updates'
            assert len(agent_messages) == 0, (
                f'Should have NO agent messages when streaming disabled, but got: {agent_messages}'
            )

            # Verify clean event stream: only status updates (working -> completed)
            status_events = [event for event in captured_events if event.get('kind') == 'status-update']
            assert len(status_events) >= 2, 'Should have at least working and completed status updates'

            # Verify final result is complete and correct
            final_result = await a2a_client.get_task(task_id)
            assert 'result' in final_result
            final_result = final_result['result']
            assert 'artifacts' in final_result
            artifacts = final_result['artifacts']
            assert len(artifacts) == 1
            artifact = cast(dict[str, Any], artifacts[0])
            assert artifact['name'] == 'result'
            assert len(artifact['parts']) == 1
            artifact_part = cast(dict[str, Any], artifact['parts'][0])
            assert artifact_part['kind'] == 'text'
            # Final result should contain all text parts concatenated
            assert 'First part of response' in artifact_part['text']


# =====================================================================
# Dependency Injection Tests
# =====================================================================


@dataclass
class MyDeps:
    """Test dependencies for dependency injection."""

    user_id: str
    auth_level: str
    custom_data: dict[str, Any]


def create_test_deps(params: TaskSendParams) -> MyDeps:
    """Factory function to create dependencies from task send params metadata."""
    metadata = params.get('metadata', {})
    return MyDeps(
        user_id=metadata.get('user_id', 'default_user'),
        auth_level=metadata.get('auth_level', 'basic'),
        custom_data=metadata.get('custom_data', {}),
    )


async def test_deps_factory_provides_dependencies():
    """Test that deps_factory correctly provides dependencies to the agent."""
    # Track what dependencies were received
    received_deps: list[MyDeps] = []

    def model_func(messages: list[ModelMessage], info: Any) -> ModelResponse:
        """Mock model that calls a tool on first request, then returns the tool result as final answer."""
        # Check if this is the first call or if we've already called the tool
        has_tool_return = any(
            isinstance(part, ToolReturnPart) and part.tool_name == 'get_user_info'
            for msg in messages
            if isinstance(msg, ModelRequest)
            for part in msg.parts
        )

        if has_tool_return:
            # Second call - we've already called the tool, return a final text response
            return ModelResponse(parts=[PydanticAITextPart(content='Tool call completed successfully')])
        else:
            # First call - request the tool
            return ModelResponse(parts=[ToolCallPart(tool_name='get_user_info', args={}, tool_call_id='1')])

    agent = Agent(model=FunctionModel(model_func), deps_type=MyDeps)

    @agent.tool
    def get_user_info(ctx: RunContext[MyDeps]) -> str:
        """Get user information from dependencies."""
        # Track the dependencies we received
        received_deps.append(ctx.deps)
        return f'User: {ctx.deps.user_id}, Auth: {ctx.deps.auth_level}'

    # Create A2A app with deps_factory
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    app = agent_to_a2a(agent, deps_factory=create_test_deps, storage=storage, broker=broker)

    async with LifespanManager(app):
        transport = ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            # Send message with metadata that will be used by deps_factory
            message = Message(
                role='user',
                parts=[TextPart(text='Get my user info', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )

            # Send with custom metadata
            response = await a2a_client.send_message(
                message=message,
                metadata={
                    'user_id': 'test_user_123',
                    'auth_level': 'admin',
                    'custom_data': {'preference': 'dark_mode'},
                },
            )

            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            task_id = result['id']

            # Wait for task completion
            task = await wait_for_task(a2a_client, task_id)

            # Debug output if task failed
            if task['status']['state'] == 'failed':
                print(f'Task failed. Full task response: {task}')
                if 'history' in task:
                    print(f'Task history: {task["history"]}')

            assert task['status']['state'] == 'completed'

            # Verify the dependencies were provided correctly
            assert len(received_deps) == 1
            deps = received_deps[0]
            assert deps.user_id == 'test_user_123'
            assert deps.auth_level == 'admin'
            assert deps.custom_data == {'preference': 'dark_mode'}

            # Check the tool was called with correct deps
            messages = task.get('history', [])
            agent_messages = [m for m in messages if m['role'] == 'agent']
            assert len(agent_messages) > 0

            # The agent should have processed the tool call and returned a final response
            # We should see "Tool call completed successfully" in the final message
            found_final_response = False
            for msg in agent_messages:
                for part in msg['parts']:
                    if part.get('kind') == 'text' and 'Tool call completed successfully' in part.get('text', ''):
                        found_final_response = True
                        break
                if found_final_response:
                    break

            assert found_final_response, 'Expected final response not found in agent messages'


async def test_deps_factory_with_no_deps():
    """Test that agents without deps_factory still work correctly."""

    def model_func(messages: list[ModelMessage], info: Any) -> ModelResponse:
        return ModelResponse(parts=[PydanticAITextPart(content='Hello from agent')])

    # Agent without deps_type
    agent = Agent(model=FunctionModel(model_func))

    storage = InMemoryStorage()
    broker = InMemoryBroker()

    # Create A2A app without deps_factory
    app = agent_to_a2a(agent, storage=storage, broker=broker)

    async with LifespanManager(app):
        transport = ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )

            response = await a2a_client.send_message(message=message)
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            task_id = result['id']

            # Wait for task completion
            task = await wait_for_task(a2a_client, task_id)
            assert task['status']['state'] == 'completed'

            # Verify agent responded
            messages = task.get('history', [])
            agent_messages = [m for m in messages if m['role'] == 'agent']
            assert len(agent_messages) > 0
            assert any('Hello from agent' in part.get('text', '') for msg in agent_messages for part in msg['parts'])


async def test_deps_factory_error_handling():
    """Test error handling when deps_factory raises an exception."""

    def failing_deps_factory(params: TaskSendParams) -> MyDeps:
        """Factory that always fails."""
        raise ValueError('Failed to create dependencies')

    def model_func(messages: list[ModelMessage], info: Any) -> ModelResponse:
        return ModelResponse(parts=[PydanticAITextPart(content='Should not reach here')])

    agent = Agent(model=FunctionModel(model_func), deps_type=MyDeps)

    storage = InMemoryStorage()
    broker = InMemoryBroker()

    # Create A2A app with failing deps_factory
    app = agent_to_a2a(agent, deps_factory=failing_deps_factory, storage=storage, broker=broker)

    async with LifespanManager(app):
        transport = ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Test message', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )

            response = await a2a_client.send_message(message=message)
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            task_id = result['id']

            # Wait for task to fail
            task = await wait_for_task(a2a_client, task_id)

            # Task should fail due to deps_factory error
            assert task['status']['state'] == 'failed'


async def test_deps_factory_type_safety():
    """Test that deps_factory maintains type safety with agent deps type."""

    @dataclass
    class SpecificDeps:
        db_connection: str
        api_key: str

    def create_specific_deps(params: TaskSendParams) -> SpecificDeps:
        metadata = params.get('metadata', {})
        return SpecificDeps(
            db_connection=metadata.get('db', 'default_db'),
            api_key=metadata.get('api_key', 'default_key'),
        )

    def model_func(messages: list[ModelMessage], info: Any) -> ModelResponse:
        """Mock model that calls a tool on first request, then returns the tool result as final answer."""
        # Check if this is the first call or if we've already called the tool
        has_tool_return = any(
            isinstance(part, ToolReturnPart) and part.tool_name == 'use_specific_deps'
            for msg in messages
            if isinstance(msg, ModelRequest)
            for part in msg.parts
        )

        if has_tool_return:
            # Second call - we've already called the tool, return a final text response
            return ModelResponse(parts=[PydanticAITextPart(content='Tool call completed successfully')])
        else:
            # First call - request the tool
            return ModelResponse(parts=[ToolCallPart(tool_name='use_specific_deps', args={}, tool_call_id='1')])

    agent = Agent(model=FunctionModel(model_func), deps_type=SpecificDeps)

    @agent.tool
    def use_specific_deps(ctx: RunContext[SpecificDeps]) -> str:
        """Use specific dependencies."""
        return f'DB: {ctx.deps.db_connection}, Key: {ctx.deps.api_key}'

    storage = InMemoryStorage()
    broker = InMemoryBroker()

    # This should type check correctly
    app = agent_to_a2a(agent, deps_factory=create_specific_deps, storage=storage, broker=broker)

    async with LifespanManager(app):
        transport = ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Use deps', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )

            response = await a2a_client.send_message(
                message=message, metadata={'db': 'production_db', 'api_key': 'secret_key'}
            )

            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            task_id = result['id']

            # Wait for task completion
            task = await wait_for_task(a2a_client, task_id)
            assert task['status']['state'] == 'completed'

            # Verify the agent completed the task with the tool call
            messages = task.get('history', [])
            agent_messages = [m for m in messages if m['role'] == 'agent']
            assert any(
                'Tool call completed successfully' in part.get('text', '')
                for msg in agent_messages
                for part in msg['parts']
            )


async def test_async_deps_factory():
    """Test that async deps_factory works correctly."""
    # Track what dependencies were received
    received_deps: list[MyDeps] = []
    factory_call_count = 0

    async def create_async_deps(params: TaskSendParams) -> MyDeps:
        """Async factory that simulates database/API calls."""
        nonlocal factory_call_count
        factory_call_count += 1

        # Simulate async I/O operation (e.g., database query)
        await anyio.sleep(0.01)

        metadata = params.get('metadata', {})
        # Simulate more complex async logic
        user_id = metadata.get('user_id', 'default_user')

        # Another async operation (e.g., fetch permissions)
        await anyio.sleep(0.01)
        auth_level = metadata.get('auth_level', 'basic')

        return MyDeps(
            user_id=user_id, auth_level=auth_level, custom_data={'async': True, 'call_number': factory_call_count}
        )

    def model_func(messages: list[ModelMessage], info: Any) -> ModelResponse:
        """Mock model that calls a tool on first request, then returns the tool result as final answer."""
        has_tool_return = any(
            isinstance(part, ToolReturnPart) and part.tool_name == 'check_async_deps'
            for msg in messages
            if isinstance(msg, ModelRequest)
            for part in msg.parts
        )

        if has_tool_return:
            return ModelResponse(parts=[PydanticAITextPart(content='Async deps test completed')])
        else:
            return ModelResponse(parts=[ToolCallPart(tool_name='check_async_deps', args={}, tool_call_id='1')])

    agent = Agent(model=FunctionModel(model_func), deps_type=MyDeps)

    @agent.tool
    def check_async_deps(ctx: RunContext[MyDeps]) -> str:
        """Tool that verifies async deps were created properly."""
        received_deps.append(ctx.deps)
        return f'Async deps received: user={ctx.deps.user_id}, async={ctx.deps.custom_data.get("async")}'

    # Create A2A app with async deps_factory
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    app = agent_to_a2a(agent, deps_factory=create_async_deps, storage=storage, broker=broker)

    async with LifespanManager(app):
        transport = ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            # Send message with metadata
            message = Message(
                role='user',
                parts=[TextPart(text='Check async deps', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )

            response = await a2a_client.send_message(
                message=message,
                metadata={
                    'user_id': 'async_test_user',
                    'auth_level': 'admin',
                },
            )

            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            task_id = result['id']

            # Wait for task completion
            task = await wait_for_task(a2a_client, task_id)
            assert task['status']['state'] == 'completed'

            # Verify async deps were created and used
            assert factory_call_count == 1
            assert len(received_deps) == 1

            deps = received_deps[0]
            assert deps.user_id == 'async_test_user'
            assert deps.auth_level == 'admin'
            assert deps.custom_data['async'] is True
            assert deps.custom_data['call_number'] == 1
