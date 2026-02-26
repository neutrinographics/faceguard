---
name: clean-architecture
description: Applies Clean Architecture and component dependency principles. Claude uses this automatically when scaffolding projects, structuring module dependencies, defining boundaries, or placing framework code.
---

# Clean Architecture Guidelines

When architecting systems, adhere strictly to the principles of separating high-level policies from low-level details.

## 1. The Dependency Rule
*   Source code dependencies must point *only* inward, toward higher-level policies.
*   Nothing in an inner circle can know anything at all about something in an outer circle.
*   The business rules should remain pristine, unsullied by baser concerns such as the user interface or database used.

## 2. The Architectural Layers (Inner to Outer)
*   **Entities (Enterprise Business Rules):** Encapsulate enterprise-wide Critical Business Rules. Independent of everything else.
*   **Use Cases (Application Business Rules):** Application-specific business rules. They orchestrate the flow of data to and from the entities. They know nothing of the UI or database.
*   **Interface Adapters:** Presenters, Views, and Controllers. They convert data from the format most convenient for the Use Cases and Entities, to the format most convenient for external agencies.
*   **Frameworks & Drivers:** The outermost layer containing the DB, web server, and third-party tools.

## 3. Crossing Boundaries
*   **Dependency Inversion Principle (DIP):** When an inner layer needs to communicate with an outer layer, it must use an interface (Port) defined in the inner layer, which the outer layer implements (Adapter). Source code dependencies oppose the flow of control.
*   **Data Transfer Across Boundaries:** Cross boundaries using isolated, simple data structures (DTOs). Never pass Entity objects or database rows across boundaries.
*   **Humble Object Pattern:** Use Humble Objects at the boundaries to separate behaviors into testable and non-testable parts (e.g., Presenters hold testable logic; Views are humble and simply display data).
*   **The Database is a Detail:** The database is a mechanism. Do not couple the architecture to the relational structure of the data.
*   **The Web is a Detail:** The web is an IO device. Your application architecture should treat it as a delivery mechanism, completely isolated from business rules.

## 4. Screaming Architecture
*   Your directory and package structure should scream the *intent* of the system (e.g., "Accounting", "HealthCare") rather than the frameworks used (e.g., "Controllers", "Views").
*   A good architecture makes it unnecessary to decide on frameworks and databases until much later in the project.

## 5. Component Principles (SOLID for Architecture)
*   **Component Cohesion:**
    *   *REP (Reuse/Release Equivalence Principle):* The granule of reuse is the granule of release.
    *   *CCP (Common Closure Principle):* Gather into components those classes that change for the same reasons and at the same times.
    *   *CRP (Common Reuse Principle):* Don't force users of a component to depend on things they don't need.
*   **Component Coupling:**
    *   *ADP (Acyclic Dependencies Principle):* Allow no cycles in the component dependency graph.
    *   *SDP (Stable Dependencies Principle):* Depend in the direction of stability.
    *   *SAP (Stable Abstractions Principle):* A component should be as abstract as it is stable.

## 6. The Main Component
*   Treat `Main` as the ultimate, dirtiest detail.
*   It is a plugin that handles object construction, Dependency Injection (e.g., Spring wiring), and configurations, then hands control over to the high-level, clean abstract portions of the system.
