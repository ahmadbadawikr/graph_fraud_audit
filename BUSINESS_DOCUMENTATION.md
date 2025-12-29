# Business Logic & Fraud Detection Model

## 1. Executive Summary
This project transforms traditional tabular financial audit data into a **Knowledge Graph** to detect complex fraud patterns that are difficult to spot with standard SQL queries. By modeling relationships between customers (`Nasabah`), employees (`Pekerja`), and their accounts (`Simpanan`, `Pinjaman`), we leverage **Graph Neural Networks (GNN)** to identify suspicious clusters and high-risk entities.

## 2. Business Entities (The "What")
The graph model consists of the following nodes, representing key business entities:

*   **Nasabah (Customer)**: Individuals or corporate entities holding accounts at the institution.
*   **Pekerja (Employee)**: Bank staff members. Identifying them explicitly is crucial for detecting **internal fraud**, **collusion**, or **conflicts of interest**.
*   **Simpanan (Savings Account)**: Deposit accounts where funds are stored.
*   **Pinjaman (Loan Account)**: Credit accounts where funds are disbursed to borrowers.
*   **Transaksi (Transaction)**: The movement of funds. In this graph model, transactions are treated as **nodes** (rather than just edges) to capture rich transactional attributes (time, amount, type) and to link multiple parties (Source $\to$ Transaction $\to$ Destination).

## 3. Key Relationships (The "How")
The edges in the graph represent the flow of ownership and funds:

### Ownership & Association
*   **Ownership**: `Nasabah` $\xrightarrow{\text{owns}}$ `Simpanan` / `Pinjaman`.
*   **Employment**: `Nasabah` $\xleftrightarrow{\text{is}}$ `Pekerja`. This "identity resolution" link helps spot if an employee is acting as a customer in suspicious ways (e.g., disbursing loans to themselves or family members).

### Money Flow (Transactional Paths)
Money does not just flow directly from A to B; it flows through a `Transaksi` event node:
1.  **Debit (Outflow)**: `Simpanan` / `Pinjaman` $\xrightarrow{\text{debit}}$ `Transaksi`
2.  **Credit (Inflow)**: `Transaksi` $\xrightarrow{\text{credit}}$ `Simpanan` / `Pinjaman`

This structure allows us to trace complex paths, such as:
> `Pinjaman` (Loan Disbursement) $\to$ `Transaksi` $\to$ `Simpanan` (Savings Account)

## 4. Fraud Scenarios Targeted
The GNN is trained to detect patterns indicative of:
*   **Employee Collusion**: Detecting if a `Pekerja` is indirectly connected to bad loans or suspicious `Nasabah` through a short chain of transactions.
*   **Circular Flows (Round-Tripping)**: Funds moving in loops (A $\to$ B $\to$ C $\to$ A) to artificially inflate turnover or hide the origin of funds (Money Laundering).
*   **Loan Layering**: Using proceeds from new loans to pay off old ones (Ponzi-like structures), often hidden through multiple pass-through accounts.
*   **Identity Fraud / Synthetic IDs**: Multiple accounts acting as distinct entities but structurally clustering around a single control point or sharing unique attributes.

## 5. Why Graph Audit?
*   **Traditional Audit**: Looks at rows in a table (e.g., *"Show me all transactions > $10,000"*). This misses context.
*   **Graph Audit**: Looks at **topology** and **neighborhoods**.
    *   *"Show me an employee connected within 2 hops to a defaulted loan account that received funds from a known high-risk external account."*
    *   *"Show me a cluster of customers who only trade with each other and have no outside interaction (Closed Loop)."*

By using a Heterogeneous Graph Neural Network (HeteroGNN), the model learns an embedding for every node that captures not just its own features (balance, age), but the **risk level of its entire neighborhood**.
