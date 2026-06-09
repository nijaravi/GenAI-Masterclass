# ============================================================
# TechStore Knowledge Base
# All policy documents as structured chunks.
# Each chunk has: id, category, title, content, keywords
#
# This is the ground truth the RAG layer retrieves from.
# In production this would be loaded from a database or
# document store — here it's a Python list for simplicity.
# ============================================================

DOCUMENTS = [
    # ── RETURNS ─────────────────────────────────────────────
    {
        "id": "ret-001",
        "category": "returns",
        "title": "Standard Return Window",
        "content": (
            "TechStore accepts returns within 15 days of delivery. "
            "Items must be in original condition with all accessories, "
            "manuals, and original packaging included. "
            "A proof of purchase (order confirmation or receipt) is required for all returns."
        ),
        "keywords": ["return", "returns", "return policy", "15 days", "how long", "window"],
    },
    {
        "id": "ret-002",
        "category": "returns",
        "title": "Non-Returnable Items",
        "content": (
            "The following items cannot be returned: opened software or digital downloads, "
            "consumables such as printer ink and batteries once opened, "
            "items marked as final sale, and products with broken seals on hygiene items. "
            "Custom-configured products (e.g. built-to-order PCs) are also non-returnable "
            "unless they arrive faulty."
        ),
        "keywords": ["cannot return", "non returnable", "no return", "final sale", "software", "digital"],
    },
    {
        "id": "ret-003",
        "category": "returns",
        "title": "Faulty or Damaged Items",
        "content": (
            "If your item arrives damaged or develops a fault within 30 days of delivery, "
            "TechStore will offer a full replacement or refund at no cost to you, "
            "including return shipping. Contact support within 30 days with photos of the damage. "
            "After 30 days, faults are handled under the standard warranty process."
        ),
        "keywords": ["faulty", "damaged", "broken", "defective", "fault", "replacement", "damaged on arrival"],
    },
    {
        "id": "ret-004",
        "category": "returns",
        "title": "Return Process — How to Return",
        "content": (
            "To initiate a return: log into your TechStore account, go to Orders, "
            "select the item and click 'Return Item'. "
            "Choose your reason and preferred resolution (refund or exchange). "
            "You will receive a prepaid return label by email within 24 hours. "
            "Drop the package at any courier point. "
            "Refunds are processed within 5–7 business days of TechStore receiving the item."
        ),
        "keywords": ["how to return", "return process", "return steps", "refund process", "return label"],
    },

    # ── SHIPPING ─────────────────────────────────────────────
    {
        "id": "ship-001",
        "category": "shipping",
        "title": "Shipping Destinations",
        "content": (
            "TechStore ships to all seven Emirates in the UAE: "
            "Abu Dhabi, Dubai, Sharjah, Ajman, Umm Al Quwain, Ras Al Khaimah, and Fujairah. "
            "International shipping is available to GCC countries: Saudi Arabia, Kuwait, "
            "Bahrain, Qatar, and Oman. "
            "Shipping to other countries is not currently supported."
        ),
        "keywords": ["ship", "shipping", "UAE", "deliver", "delivery", "Abu Dhabi", "Dubai", "GCC", "international"],
    },
    {
        "id": "ship-002",
        "category": "shipping",
        "title": "Shipping Cost and Free Shipping Threshold",
        "content": (
            "Standard shipping within UAE costs AED 15 for orders below AED 100. "
            "Orders of AED 100 or more qualify for free standard shipping. "
            "Express shipping (next business day) is available for AED 35 regardless of order value. "
            "GCC international shipping starts at AED 45 and varies by destination and weight."
        ),
        "keywords": ["shipping cost", "delivery fee", "free shipping", "AED", "express", "how much"],
    },
    {
        "id": "ship-003",
        "category": "shipping",
        "title": "Delivery Timeframes",
        "content": (
            "Standard delivery within UAE: 2–3 business days for major cities (Abu Dhabi, Dubai, Sharjah), "
            "3–5 business days for other Emirates. "
            "Express delivery: next business day if ordered before 2 PM. "
            "GCC countries: 4–7 business days. "
            "Delivery times may be longer during public holidays and peak sale periods."
        ),
        "keywords": ["delivery time", "how long", "when will arrive", "business days", "express", "next day"],
    },
    {
        "id": "ship-004",
        "category": "shipping",
        "title": "Order Tracking",
        "content": (
            "Once your order is dispatched, you will receive an SMS and email with a tracking number. "
            "Track your order at techstore.ae/track or through the TechStore mobile app. "
            "Live tracking updates are provided every time the package moves between courier facilities. "
            "If your order has not moved for more than 48 hours, contact support with your tracking number."
        ),
        "keywords": ["track", "tracking", "where is my order", "order status", "tracking number"],
    },

    # ── PRODUCTS ─────────────────────────────────────────────
    {
        "id": "prod-001",
        "category": "products",
        "title": "TechStore Pro X — Key Specifications",
        "content": (
            "The TechStore Pro X is a flagship smartphone with the following specifications: "
            "6.7-inch AMOLED display at 120Hz, Snapdragon 8 Gen 3 processor, "
            "12GB RAM, 256GB or 512GB storage options. "
            "Battery: 5000mAh with 65W fast charging and 15W Qi wireless charging. "
            "Camera: 50MP main, 12MP ultrawide, 10MP telephoto (3× optical zoom). "
            "IP68 water and dust resistance. Available in Midnight Black, Arctic White, and Deep Blue."
        ),
        "keywords": ["Pro X", "specifications", "specs", "wireless charging", "battery", "camera", "storage"],
    },
    {
        "id": "prod-002",
        "category": "products",
        "title": "TechStore Laptop Range — Warranty",
        "content": (
            "All TechStore laptops come with a 2-year manufacturer warranty covering hardware defects. "
            "The warranty does not cover physical damage, liquid damage, or damage from unauthorised repairs. "
            "An optional TechStore Care+ extended warranty adds 1 additional year and covers accidental damage. "
            "Care+ can be purchased within 30 days of buying the laptop for AED 299."
        ),
        "keywords": ["laptop", "warranty", "2 year", "Care+", "extended warranty", "accidental damage"],
    },
    {
        "id": "prod-003",
        "category": "products",
        "title": "TechStore Pro X — Wireless Charging",
        "content": (
            "The TechStore Pro X supports Qi wireless charging at up to 15W. "
            "It is compatible with any Qi-certified wireless charger. "
            "For best results, use the TechStore 15W Wireless Charging Pad (sold separately, AED 89). "
            "The phone also supports reverse wireless charging at 5W, allowing it to charge "
            "compatible accessories like wireless earbuds placed on the back of the phone."
        ),
        "keywords": ["wireless charging", "Qi", "15W", "charger", "Pro X", "charging pad"],
    },
    {
        "id": "prod-004",
        "category": "products",
        "title": "Product Compatibility and Accessories",
        "content": (
            "TechStore accessories are designed for use with TechStore devices but follow open standards. "
            "USB-C cables and chargers from TechStore work with any USB-C device. "
            "TechStore cases are model-specific — check the product page for compatibility. "
            "The TechStore 15W Wireless Pad works with any Qi-certified device including Apple and Samsung phones."
        ),
        "keywords": ["accessories", "compatible", "compatibility", "USB-C", "case", "works with"],
    },

    # ── PAYMENT ──────────────────────────────────────────────
    {
        "id": "pay-001",
        "category": "payment",
        "title": "Accepted Payment Methods",
        "content": (
            "TechStore accepts the following payment methods: "
            "Visa and Mastercard credit and debit cards, "
            "Apple Pay and Google Pay, "
            "Cash on Delivery (COD) for UAE orders up to AED 2,000, "
            "TechStore Gift Cards, "
            "and bank transfer for orders above AED 5,000. "
            "Cryptocurrency payments are not accepted."
        ),
        "keywords": ["payment", "pay", "credit card", "debit card", "Apple Pay", "cash on delivery", "COD", "cryptocurrency"],
    },
    {
        "id": "pay-002",
        "category": "payment",
        "title": "Payment Security",
        "content": (
            "TechStore uses PCI-DSS Level 1 compliant payment processing. "
            "Card details are encrypted during transmission using TLS 1.3. "
            "TechStore does not store full card numbers — only the last 4 digits are retained for reference. "
            "All transactions are protected by 3D Secure authentication where supported by your bank."
        ),
        "keywords": ["safe", "secure", "security", "card", "PCI", "encrypted", "store card details"],
    },
    {
        "id": "pay-003",
        "category": "payment",
        "title": "Buy Now Pay Later — Tabby and Tamara",
        "content": (
            "TechStore offers Buy Now Pay Later (BNPL) through Tabby and Tamara. "
            "Split any purchase into 4 equal interest-free payments. "
            "Available for orders between AED 200 and AED 10,000. "
            "Select Tabby or Tamara at checkout. Approval is subject to the provider's eligibility check. "
            "BNPL is available for UAE residents only."
        ),
        "keywords": ["installment", "BNPL", "buy now pay later", "Tabby", "Tamara", "split payment", "pay later"],
    },

    # ── ACCOUNT ──────────────────────────────────────────────
    {
        "id": "acc-001",
        "category": "account",
        "title": "Password Reset",
        "content": (
            "To reset your TechStore account password: "
            "go to techstore.ae/login and click 'Forgot Password'. "
            "Enter your registered email address. "
            "A password reset link will be sent to your email within 2 minutes. "
            "The link is valid for 24 hours. "
            "If you don't receive the email, check your spam folder or contact support."
        ),
        "keywords": ["password", "reset password", "forgot password", "login", "can't login", "account access"],
    },
    {
        "id": "acc-002",
        "category": "account",
        "title": "Saved Delivery Addresses",
        "content": (
            "You can save up to 5 delivery addresses in your TechStore account. "
            "To manage addresses: go to Account Settings > Delivery Addresses. "
            "You can add, edit, or delete saved addresses at any time. "
            "Set a default address for faster checkout. "
            "Each address can be labelled (e.g. Home, Work, Parents)."
        ),
        "keywords": ["address", "delivery address", "multiple addresses", "save address", "default address"],
    },
    {
        "id": "acc-003",
        "category": "account",
        "title": "Order History and Invoices",
        "content": (
            "Your full order history is available under Account > My Orders. "
            "Each order shows the status, items, delivery information, and a downloadable PDF invoice. "
            "Invoices include VAT breakdown as required by UAE Federal Tax Authority regulations. "
            "Orders are retained in your history for 3 years. "
            "For orders placed as a guest, use the Order Lookup tool with your email and order number."
        ),
        "keywords": ["order history", "invoice", "VAT", "receipt", "past orders", "download invoice"],
    },

    # ── WARRANTY & REPAIRS ────────────────────────────────────
    {
        "id": "war-001",
        "category": "warranty",
        "title": "Standard Warranty Coverage",
        "content": (
            "All TechStore products include a minimum 1-year manufacturer warranty. "
            "Laptops and high-value electronics come with a 2-year warranty. "
            "The warranty covers manufacturing defects and hardware failures under normal use. "
            "It does not cover: physical damage, water damage, damage from power surges, "
            "or modifications by unauthorized repair centres."
        ),
        "keywords": ["warranty", "covered", "not covered", "1 year", "2 year", "manufacturer warranty"],
    },
    {
        "id": "war-002",
        "category": "warranty",
        "title": "How to Claim Warranty",
        "content": (
            "To make a warranty claim: contact TechStore support via chat or phone. "
            "Provide your order number, product serial number, and a description of the fault. "
            "TechStore will arrange collection of the item for inspection. "
            "If confirmed as a manufacturing defect, the item will be repaired or replaced at no charge. "
            "Warranty claims are typically resolved within 7–10 business days."
        ),
        "keywords": ["warranty claim", "how to claim", "repair", "warranty process", "defect", "serial number"],
    },
]
