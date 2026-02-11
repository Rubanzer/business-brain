"""Seed sample metadata entries matching the PRD example."""

import asyncio

from business_brain.db.connection import async_session, engine
from business_brain.memory import metadata_store

SAMPLE_ENTRIES = [
    {
        "table_name": "sales_orders",
        "description": "Transactional table storing all customer sales orders with line items, dates, and amounts.",
        "columns_metadata": [
            {"name": "order_id", "type": "integer", "description": "Primary key"},
            {"name": "customer_id", "type": "integer", "description": "FK to customers table"},
            {"name": "order_date", "type": "date", "description": "Date the order was placed"},
            {"name": "total_amount", "type": "numeric", "description": "Total order value"},
            {"name": "status", "type": "varchar", "description": "Order status: pending/shipped/delivered"},
        ],
    },
    {
        "table_name": "customers",
        "description": "Master table of all registered customers with contact and segment info.",
        "columns_metadata": [
            {"name": "customer_id", "type": "integer", "description": "Primary key"},
            {"name": "name", "type": "varchar", "description": "Customer full name"},
            {"name": "segment", "type": "varchar", "description": "Market segment: SMB/Enterprise/Consumer"},
            {"name": "region", "type": "varchar", "description": "Geographic region"},
        ],
    },
    {
        "table_name": "products",
        "description": "Product catalog with pricing and category information.",
        "columns_metadata": [
            {"name": "product_id", "type": "integer", "description": "Primary key"},
            {"name": "name", "type": "varchar", "description": "Product name"},
            {"name": "category", "type": "varchar", "description": "Product category"},
            {"name": "unit_price", "type": "numeric", "description": "Price per unit"},
        ],
    },
]


async def seed() -> None:
    async with async_session() as session:
        for entry in SAMPLE_ENTRIES:
            await metadata_store.upsert(
                session,
                table_name=entry["table_name"],
                description=entry["description"],
                columns_metadata=entry["columns_metadata"],
            )
            print(f"[seed] Upserted metadata for: {entry['table_name']}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())
