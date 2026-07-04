"""
Sample content for the two namespaces described in Section 8.1: customer
product data and developer reference docs, sharing one retrieval layer but
never mixing at query time.
"""

CUSTOMER_PRODUCT_DOCS = [
    {
        "id": "prod-101",
        "text": "The TrailPro Hiking Backpack is available in three sizes: "
                "Small (35L), Medium (45L), and Large (55L). The Large size "
                "was added in the Spring 2026 refresh for taller frames.",
        "source": "catalog/trailpro-backpack.md",
    },
    {
        "id": "prod-102",
        "text": "The TrailPro Hiking Backpack ships in Slate Grey, Forest "
                "Green, and Rust Orange. All colors are available in every size.",
        "source": "catalog/trailpro-backpack.md",
    },
    {
        "id": "prod-103",
        "text": "Returns on the TrailPro line are accepted within 60 days "
                "of delivery, provided the item is unused and tags are attached.",
        "source": "catalog/trailpro-backpack.md",
    },
    {
        "id": "prod-104",
        "text": "The AeroFlow Running Shoe comes in Men's sizes 7-13 and "
                "Women's sizes 5-11, in half-size increments.",
        "source": "catalog/aeroflow-shoe.md",
    },
]

DEVELOPER_DOCS = [
    {
        "id": "dev-201",
        "text": "To call the internal Auth API, POST to /v1/auth/token with "
                "client_id and client_secret in the body. The response contains "
                "a bearer token valid for 1 hour. All internal services require "
                "this token in the Authorization header.",
        "source": "docs/internal-auth-api.md",
    },
    {
        "id": "dev-202",
        "text": "Rate limits on the Auth API are 100 requests/minute per "
                "client_id. Exceeding this returns HTTP 429 with a Retry-After header.",
        "source": "docs/internal-auth-api.md",
    },
    {
        "id": "dev-203",
        "text": "The Orders service exposes GET /v1/orders/{order_id} and "
                "requires the 'orders:read' scope on the caller's token.",
        "source": "docs/orders-api.md",
    },
]
