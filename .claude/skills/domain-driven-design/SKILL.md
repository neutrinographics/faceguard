---
name: domain-driven-design
description: Applies Domain-Driven Design (DDD) tactical and strategic patterns. Claude uses this automatically when modeling core business rules, creating domain objects, or defining system boundaries and context maps.
---

# Domain-Driven Design Guidelines

When creating business logic or domain models, act as a domain expert and apply both tactical and strategic DDD patterns:

## 1. Ubiquitous Language & Domain Isolation
*   **Ubiquitous Language:** Use the vocabulary of the domain explicitly in class names, methods, and variables. Play with the model as you talk about the system. If the language changes, the code must change.
*   **Layered Architecture:** Concentrate all the code related to the domain model in one layer and isolate it from the user interface, application, and infrastructure code. The domain layer is where the software expression of the domain model lives.

## 2. Tactical Patterns (Expressing the Model)
*   **Entities:** Objects fundamentally defined not by their attributes, but by a thread of continuity and identity.
*   **Value Objects:** Immutable objects that describe some characteristic or attribute but carry no concept of identity. Use these to represent descriptive aspects of the domain.
*   **Domain Services:** When a significant process or transformation in the domain is not a natural responsibility of an Entity or Value Object, add an operation to the model as a standalone interface declared as a Service. Make the Service stateless.
*   **Modules:** Choose modules that tell the story of the system and contain a cohesive set of concepts.

## 3. The Life Cycle of a Domain Object
*   **Aggregates:** Cluster Entities and Value Objects into Aggregates and define boundaries around each. Choose one Entity to be the root of each Aggregate, and control all access to the objects inside the boundary through the root. The Aggregate must maintain its invariants at every stage of the life cycle.
*   **Factories:** Use factories to encapsulate the complex creation and reconstitution of Aggregates, keeping the domain objects clean of creation logic.
*   **Repositories:** For each type of object that needs global access, create an object that can provide the illusion of an in-memory collection. Repositories encapsulate database access technology and provide access *only* to Aggregate roots.

## 4. Strategic Design (Contexts & Integration)
In large systems, it is impossible to maintain a single, unified model. You must consciously define boundaries and integration strategies.

*   **Bounded Context:** The delimited applicability of a particular model. Keep the model strictly consistent within these bounds, but do not worry about its applicability outside of them.
*   **Context Map:** A representation of the Bounded Contexts involved in a project and the actual relationships between them.
*   **Integration Patterns:**
    *   *Shared Kernel:* A highly distilled subset of the domain model shared between two teams.
    *   *Customer/Supplier Teams:* An upstream/downstream relationship where the supplier (upstream) accommodates the customer (downstream).
    *   *Conformist:* When the downstream team slavishly adheres to the upstream team's model.
    *   *Anticorruption Layer (ACL):* Create an isolating translation layer that talks to an external system but provides clients with functionality strictly in terms of the clean domain model.
    *   *Open Host Service / Published Language:* Define a protocol/API and a well-documented shared language (e.g., XML/JSON schema) as a common medium of communication.
    *   *Separate Ways:* If integration provides no significant benefit, declare the Bounded Contexts to have no connection.

## 5. Distillation
*   **Core Domain:** The distinctive part of the model, central to the user's goals, that differentiates the application and makes it valuable. Make the Core Domain small and apply top talent to it.
*   **Generic Subdomains:** Identify cohesive subdomains that are not the motivation for your project (e.g., standard accounting, timezones) and factor them out.
*   **Cohesive Mechanisms:** Partition a conceptually cohesive mechanism (complex algorithms) into a separate lightweight framework to unburden the Core Domain.
