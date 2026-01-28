"""
Message queue for real-time data flow.

Supports both in-memory queue (development) and Redis (production).
"""

import asyncio
import json
import logging

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueueMessage:
    """Message wrapper for queue."""
    channel: str
    data: Dict[str, Any]
    timestamp: datetime
    
    def to_json(self) -> str:
        return json.dumps({
            "channel": self.channel,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> "QueueMessage":
        obj = json.loads(json_str)
        return cls(
            channel=obj["channel"],
            data=obj["data"],
            timestamp=datetime.fromisoformat(obj["timestamp"]),
        )


class MessageQueueBase(ABC):
    """Abstract base class for message queue implementations."""
    
    @abstractmethod
    async def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish a message to a channel."""
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        channel: str,
        callback: Callable[[QueueMessage], None]
    ) -> None:
        """Subscribe to a channel with a callback."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from a channel."""
        pass
    
    @abstractmethod
    async def get_batch(
        self,
        channel: str,
        max_messages: int = 100,
        timeout_ms: int = 100
    ) -> List[QueueMessage]:
        """Get a batch of messages from a channel."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the queue connection."""
        pass


class InMemoryQueue(MessageQueueBase):
    """
    In-memory message queue for development and testing.
    
    Thread-safe using asyncio locks.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queues: Dict[str, deque] = {}
        self._subscribers: Dict[str, List[Callable]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._running = True
        self._dispatcher_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the message dispatcher."""
        self._running = True
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        logger.info("InMemoryQueue dispatcher started")
    
    async def _dispatch_loop(self):
        """Background loop to dispatch messages to subscribers."""
        while self._running:
            for channel, callbacks in list(self._subscribers.items()):
                if channel in self._queues and self._queues[channel]:
                    async with self._get_lock(channel):
                        while self._queues[channel]:
                            msg = self._queues[channel].popleft()
                            for callback in callbacks:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(msg)
                                    else:
                                        callback(msg)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")
            
            await asyncio.sleep(0.01)  # 10ms sleep between dispatch cycles
    
    def _get_lock(self, channel: str) -> asyncio.Lock:
        """Get or create lock for channel."""
        if channel not in self._locks:
            self._locks[channel] = asyncio.Lock()
        return self._locks[channel]
    
    async def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish message to channel."""
        if channel not in self._queues:
            self._queues[channel] = deque(maxlen=self.max_size)
        
        queue_msg = QueueMessage(
            channel=channel,
            data=message,
            timestamp=datetime.utcnow()
        )
        
        async with self._get_lock(channel):
            if len(self._queues[channel]) >= self.max_size:
                # Drop oldest message
                self._queues[channel].popleft()
                logger.warning(f"Queue {channel} full, dropping oldest message")
            
            self._queues[channel].append(queue_msg)
        
        return True
    
    async def subscribe(
        self,
        channel: str,
        callback: Callable[[QueueMessage], None]
    ) -> None:
        """Subscribe to channel with callback."""
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        self._subscribers[channel].append(callback)
        logger.debug(f"Subscribed to channel: {channel}")
    
    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from channel."""
        if channel in self._subscribers:
            del self._subscribers[channel]
        logger.debug(f"Unsubscribed from channel: {channel}")
    
    async def get_batch(
        self,
        channel: str,
        max_messages: int = 100,
        timeout_ms: int = 100
    ) -> List[QueueMessage]:
        """Get batch of messages without using callback."""
        messages = []
        
        if channel not in self._queues:
            return messages
        
        start_time = datetime.utcnow()
        timeout = timeout_ms / 1000
        
        while len(messages) < max_messages:
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed >= timeout:
                break
            
            async with self._get_lock(channel):
                if self._queues[channel]:
                    messages.append(self._queues[channel].popleft())
                else:
                    await asyncio.sleep(0.001)  # 1ms
        
        return messages
    
    async def close(self) -> None:
        """Close the queue."""
        self._running = False
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
        logger.info("InMemoryQueue closed")
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current size of all queues."""
        return {channel: len(q) for channel, q in self._queues.items()}


class RedisQueue(MessageQueueBase):
    """
    Redis-based message queue for production.
    
    Uses Redis Pub/Sub for real-time messaging and Lists for batch processing.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self._redis = None
        self._pubsub = None
        self._subscriber_tasks: Dict[str, asyncio.Task] = {}
    
    async def connect(self):
        """Connect to Redis."""
        try:
            import redis.asyncio as aioredis
            
            self._redis = aioredis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            self._pubsub = self._redis.pubsub()
            
            # Test connection
            await self._redis.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            
        except ImportError:
            raise ImportError(
                "redis package required for RedisQueue. "
                "Install with: pip install redis"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish message to Redis channel."""
        if not self._redis:
            await self.connect()
        
        try:
            queue_msg = QueueMessage(
                channel=channel,
                data=message,
                timestamp=datetime.utcnow()
            )
            
            # Publish to pub/sub for real-time subscribers
            await self._redis.publish(channel, queue_msg.to_json())
            
            # Also push to list for batch consumers
            list_key = f"queue:{channel}"
            await self._redis.lpush(list_key, queue_msg.to_json())
            
            # Trim list to prevent unbounded growth
            await self._redis.ltrim(list_key, 0, 9999)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}")
            return False
    
    async def subscribe(
        self,
        channel: str,
        callback: Callable[[QueueMessage], None]
    ) -> None:
        """Subscribe to Redis channel."""
        if not self._redis:
            await self.connect()
        
        await self._pubsub.subscribe(channel)
        
        async def listener():
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    try:
                        queue_msg = QueueMessage.from_json(message["data"])
                        if asyncio.iscoroutinefunction(callback):
                            await callback(queue_msg)
                        else:
                            callback(queue_msg)
                    except Exception as e:
                        logger.error(f"Callback error for {channel}: {e}")
        
        task = asyncio.create_task(listener())
        self._subscriber_tasks[channel] = task
        logger.info(f"Subscribed to Redis channel: {channel}")
    
    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from Redis channel."""
        await self._pubsub.unsubscribe(channel)
        
        if channel in self._subscriber_tasks:
            self._subscriber_tasks[channel].cancel()
            try:
                await self._subscriber_tasks[channel]
            except asyncio.CancelledError:
                pass
            del self._subscriber_tasks[channel]
        
        logger.info(f"Unsubscribed from Redis channel: {channel}")
    
    async def get_batch(
        self,
        channel: str,
        max_messages: int = 100,
        timeout_ms: int = 100
    ) -> List[QueueMessage]:
        """Get batch of messages from Redis list."""
        if not self._redis:
            await self.connect()
        
        messages = []
        list_key = f"queue:{channel}"
        
        try:
            # Use RPOP to get oldest messages first (FIFO)
            for _ in range(max_messages):
                data = await self._redis.rpop(list_key)
                if data is None:
                    break
                messages.append(QueueMessage.from_json(data))
        
        except Exception as e:
            logger.error(f"Failed to get batch from {channel}: {e}")
        
        return messages
    
    async def close(self) -> None:
        """Close Redis connection."""
        for task in self._subscriber_tasks.values():
            task.cancel()
        
        if self._pubsub:
            await self._pubsub.close()
        
        if self._redis:
            await self._redis.close()
        
        logger.info("Redis connection closed")


def create_queue(
    queue_type: str = "memory",
    **kwargs
) -> MessageQueueBase:
    """
    Factory function to create appropriate queue implementation.
    
    Args:
        queue_type: "memory" or "redis"
        **kwargs: Additional arguments for the queue implementation
    
    Returns:
        MessageQueueBase instance
    """
    if queue_type == "memory":
        return InMemoryQueue(**kwargs)
    elif queue_type == "redis":
        return RedisQueue(**kwargs)
    else:
        raise ValueError(f"Unknown queue type: {queue_type}")


# Channel names for the data pipeline
class Channels:
    """Standard channel names for the data pipeline."""
    
    # Raw data channels (from connectors)
    RAW_OHLCV = "raw:ohlcv"
    RAW_TRADES = "raw:trades"
    RAW_TICKER = "raw:ticker"
    RAW_ORDERBOOK = "raw:orderbook"
    RAW_FUNDING = "raw:funding"
    RAW_OPEN_INTEREST = "raw:oi"
    
    # Validated data channels (after validation)
    VALIDATED_OHLCV = "validated:ohlcv"
    VALIDATED_TRADES = "validated:trades"
    VALIDATED_TICKER = "validated:ticker"
    VALIDATED_ORDERBOOK = "validated:orderbook"
    VALIDATED_FUNDING = "validated:funding"
    VALIDATED_OPEN_INTEREST = "validated:oi"
    
    # Error/alert channels
    VALIDATION_ERRORS = "alerts:validation"
    CONNECTION_ERRORS = "alerts:connection"
    
    @classmethod
    def raw_for_type(cls, data_type: str) -> str:
        """Get raw channel for data type."""
        return f"raw:{data_type}"
    
    @classmethod
    def validated_for_type(cls, data_type: str) -> str:
        """Get validated channel for data type."""
        return f"validated:{data_type}"
