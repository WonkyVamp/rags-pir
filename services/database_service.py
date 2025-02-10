from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
import logging
from enum import Enum
import motor.motor_asyncio
from pymongo import IndexModel, ASCENDING, DESCENDING
import asyncpg
from dataclasses import dataclass
import json
from contextlib import asynccontextmanager


class DatabaseType(Enum):
    MONGODB = "mongodb"
    POSTGRESQL = "postgresql"


class CollectionName(Enum):
    TRANSACTIONS = "transactions"
    CUSTOMERS = "customers"
    ALERTS = "alerts"
    RISK_SCORES = "risk_scores"
    PATTERNS = "patterns"
    AUDIT_LOGS = "audit_logs"


@dataclass
class DatabaseConfig:
    db_type: DatabaseType
    host: str
    port: int
    database: str
    username: str
    password: str
    min_pool_size: int = 5
    max_pool_size: int = 20
    connection_timeout: int = 5000


class DatabaseService:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = self._setup_logger()
        self._client = None
        self._pool = None
        self.connected = False
        self._setup_indices = {
            CollectionName.TRANSACTIONS: [
                IndexModel([("transaction_id", ASCENDING)], unique=True),
                IndexModel([("customer_id", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("risk_score", DESCENDING)]),
                IndexModel([("status", ASCENDING)]),
            ],
            CollectionName.CUSTOMERS: [
                IndexModel([("customer_id", ASCENDING)], unique=True),
                IndexModel([("risk_level", DESCENDING)]),
                IndexModel([("email", ASCENDING)], unique=True),
            ],
            CollectionName.ALERTS: [
                IndexModel([("alert_id", ASCENDING)], unique=True),
                IndexModel([("customer_id", ASCENDING), ("created_at", DESCENDING)]),
                IndexModel([("status", ASCENDING), ("priority", DESCENDING)]),
            ],
            CollectionName.RISK_SCORES: [
                IndexModel([("customer_id", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("score", DESCENDING)]),
            ],
        }

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("database_service")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    async def connect(self):
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                await self._connect_mongodb()
            else:
                await self._connect_postgresql()
            self.connected = True
            self.logger.info(f"Connected to {self.config.db_type.value} database")
        except Exception as e:
            self.logger.error(f"Database connection failed: {str(e)}")
            raise

    async def _connect_mongodb(self):
        connection_string = (
            f"mongodb://{self.config.username}:{self.config.password}@"
            f"{self.config.host}:{self.config.port}"
        )
        self._client = motor.motor_asyncio.AsyncIOMotorClient(
            connection_string,
            minPoolSize=self.config.min_pool_size,
            maxPoolSize=self.config.max_pool_size,
            serverSelectionTimeoutMS=self.config.connection_timeout,
        )
        self.db = self._client[self.config.database]
        await self._setup_mongodb_indices()

    async def _connect_postgresql(self):
        self._pool = await asyncpg.create_pool(
            user=self.config.username,
            password=self.config.password,
            database=self.config.database,
            host=self.config.host,
            port=self.config.port,
            min_size=self.config.min_pool_size,
            max_size=self.config.max_pool_size,
        )
        await self._setup_postgresql_tables()

    async def _setup_mongodb_indices(self):
        for collection_name, indices in self._setup_indices.items():
            collection = self.db[collection_name.value]
            await collection.create_indexes(indices)

    async def _setup_postgresql_tables(self):
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id VARCHAR(50) PRIMARY KEY,
                    customer_id VARCHAR(50) NOT NULL,
                    amount DECIMAL(15,2) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    risk_score FLOAT,
                    data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id VARCHAR(50) PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    risk_level VARCHAR(20) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id VARCHAR(50) PRIMARY KEY,
                    customer_id VARCHAR(50) NOT NULL,
                    priority VARCHAR(20) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS risk_scores (
                    id SERIAL PRIMARY KEY,
                    customer_id VARCHAR(50) NOT NULL,
                    score FLOAT NOT NULL,
                    factors JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
            )

    @asynccontextmanager
    async def transaction(self):
        if self.config.db_type == DatabaseType.MONGODB:
            async with await self._client.start_session() as session:
                async with session.start_transaction():
                    yield session
        else:
            async with self._pool.acquire() as connection:
                async with connection.transaction():
                    yield connection

    async def insert_one(
        self, collection: CollectionName, document: Dict[str, Any]
    ) -> str:
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                result = await self.db[collection.value].insert_one(document)
                return str(result.inserted_id)
            else:
                table = collection.value
                columns = ", ".join(document.keys())
                values = ", ".join(f"${i}" for i in range(1, len(document) + 1))
                query = (
                    f"INSERT INTO {table} ({columns}) VALUES ({values}) RETURNING id"
                )
                async with self._pool.acquire() as conn:
                    result = await conn.fetchval(query, *document.values())
                return str(result)
        except Exception as e:
            self.logger.error(f"Insert failed: {str(e)}")
            raise

    async def find_one(
        self, collection: CollectionName, query: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                return await self.db[collection.value].find_one(query)
            else:
                table = collection.value
                conditions = " AND ".join(
                    f"{k} = ${i}" for i, k in enumerate(query.keys(), 1)
                )
                query_str = f"SELECT * FROM {table} WHERE {conditions} LIMIT 1"
                async with self._pool.acquire() as conn:
                    result = await conn.fetchrow(query_str, *query.values())
                    return dict(result) if result else None
        except Exception as e:
            self.logger.error(f"Find one failed: {str(e)}")
            raise

    async def find_many(
        self,
        collection: CollectionName,
        query: Dict[str, Any],
        sort: Optional[List[tuple]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                cursor = self.db[collection.value].find(query)
                if sort:
                    cursor = cursor.sort(sort)
                if limit:
                    cursor = cursor.limit(limit)
                return await cursor.to_list(length=None)
            else:
                table = collection.value
                conditions = " AND ".join(
                    f"{k} = ${i}" for i, k in enumerate(query.keys(), 1)
                )
                query_str = f"SELECT * FROM {table} WHERE {conditions}"
                if sort:
                    query_str += " ORDER BY " + ", ".join(
                        f"{field} {order}" for field, order in sort
                    )
                if limit:
                    query_str += f" LIMIT {limit}"
                async with self._pool.acquire() as conn:
                    results = await conn.fetch(query_str, *query.values())
                    return [dict(r) for r in results]
        except Exception as e:
            self.logger.error(f"Find many failed: {str(e)}")
            raise

    async def update_one(
        self, collection: CollectionName, query: Dict[str, Any], update: Dict[str, Any]
    ) -> bool:
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                result = await self.db[collection.value].update_one(
                    query, {"$set": update}
                )
                return result.modified_count > 0
            else:
                table = collection.value
                set_values = ", ".join(
                    f"{k} = ${i}" for i, k in enumerate(update.keys(), 1)
                )
                where_values = " AND ".join(
                    f"{k} = ${i}" for i, k in enumerate(query.keys(), len(update) + 1)
                )
                query_str = f"UPDATE {table} SET {set_values} WHERE {where_values}"
                async with self._pool.acquire() as conn:
                    result = await conn.execute(
                        query_str, *update.values(), *query.values()
                    )
                    return result[-1] > 0
        except Exception as e:
            self.logger.error(f"Update one failed: {str(e)}")
            raise

    async def delete_one(
        self, collection: CollectionName, query: Dict[str, Any]
    ) -> bool:
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                result = await self.db[collection.value].delete_one(query)
                return result.deleted_count > 0
            else:
                table = collection.value
                conditions = " AND ".join(
                    f"{k} = ${i}" for i, k in enumerate(query.keys(), 1)
                )
                query_str = f"DELETE FROM {table} WHERE {conditions}"
                async with self._pool.acquire() as conn:
                    result = await conn.execute(query_str, *query.values())
                    return result[-1] > 0
        except Exception as e:
            self.logger.error(f"Delete one failed: {str(e)}")
            raise

    async def aggregate(
        self, collection: CollectionName, pipeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                return await self.db[collection.value].aggregate(pipeline).to_list(None)
            else:
                # Convert MongoDB aggregation pipeline to PostgreSQL
                raise NotImplementedError(
                    "PostgreSQL aggregation pipeline conversion not implemented"
                )
        except Exception as e:
            self.logger.error(f"Aggregation failed: {str(e)}")
            raise

    async def count(self, collection: CollectionName, query: Dict[str, Any]) -> int:
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                return await self.db[collection.value].count_documents(query)
            else:
                table = collection.value
                conditions = " AND ".join(
                    f"{k} = ${i}" for i, k in enumerate(query.keys(), 1)
                )
                query_str = f"SELECT COUNT(*) FROM {table} WHERE {conditions}"
                async with self._pool.acquire() as conn:
                    return await conn.fetchval(query_str, *query.values())
        except Exception as e:
            self.logger.error(f"Count failed: {str(e)}")
            raise

    async def create_index(
        self, collection: CollectionName, keys: List[tuple], unique: bool = False
    ):
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                await self.db[collection.value].create_index(keys, unique=unique)
            else:
                table = collection.value
                index_name = f"idx_{table}_{'_'.join(k[0] for k in keys)}"
                index_fields = ", ".join(f"{k[0]} {k[1]}" for k in keys)
                query = f"CREATE {'UNIQUE ' if unique else ''}INDEX IF NOT EXISTS {index_name} ON {table} ({index_fields})"
                async with self._pool.acquire() as conn:
                    await conn.execute(query)
        except Exception as e:
            self.logger.error(f"Index creation failed: {str(e)}")
            raise

    async def close(self):
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                self._client.close()
            else:
                await self._pool.close()
            self.connected = False
            self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Connection close failed: {str(e)}")
            raise
