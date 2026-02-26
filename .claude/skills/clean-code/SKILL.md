---
name: clean-code
description: Applies Clean Code, software craftsmanship, and TDD principles. Claude uses this automatically when writing, refactoring, or reviewing functions, classes, data structures, and test suites.
---

# Clean Code & Craftsmanship Guidelines

You are a professional software craftsman. Apply the following rules when writing or modifying code. The goal is to write clean code that looks like it was written by someone who cares.

## The Boy Scout Rule & DRY
*   **Leave the campground cleaner than you found it:** Always check in a module cleaner than when you checked it out.
*   **DRY (Don't Repeat Yourself):** Duplication is the primary enemy of a well-designed system. Every piece of knowledge must have a single, unambiguous, authoritative representation within a system.

## Naming
*   **Intention-Revealing:** Names must answer why a variable exists, what it does, and how it is used.
*   **Pronounceable & Searchable:** Humans are good at words; use pronounceable names. Any searchable name trumps a constant in code.
*   **No Encodings:** Do not encode type or scope information into names. Avoid Hungarian notation or member prefixes (like `m_`).
*   **Classes vs. Methods:** Classes and objects should have noun or noun phrase names. Methods should have verb or verb phrase names.
*   **One Word per Concept:** Pick one word for one abstract concept and stick with it. Don't mix `fetch`, `retrieve`, and `get`. Don't pun by using the same word for two different purposes.

## Functions
*   **Small & Focused:** The first rule of functions is that they should be small; the second rule is that they should be smaller than that. They should hardly ever be 20 lines long.
*   **Do One Thing:** Functions should do one thing, do it well, and do it only.
*   **One Level of Abstraction:** The statements within a function should all be at the same level of abstraction, which should be one level below the operation described by the name of the function.
*   **The Step-down Rule:** Code should read like a top-down narrative. Every function should be followed by those at the next level of abstraction.
*   **Arguments:** The ideal number of arguments is zero (niladic), followed by one (monadic), and two (dyadic). Three should be avoided. Do not use boolean flag arguments; they violate the rule of doing one thing.
*   **Switch Statements:** Switch statements should be tolerated only if they appear once, are used to create polymorphic objects, and are hidden behind an inheritance relationship.
*   **Command Query Separation:** Functions should either change the state of an object or return information about it, never both.

## Comments
*   **Code as Explanation:** Clear and expressive code with few comments is far superior to cluttered and complex code with lots of comments.
*   **Delete Bad Comments:** Ruthlessly delete commented-out code, redundant comments, journal/history comments, and misleading comments.
*   **Good Comments:** Only comment to explain *why* a decision was made (intent), to warn of consequences, or to amplify the importance of a seemingly inconsequential detail.

## Objects vs. Data Structures
*   **Data Abstraction:** Objects hide their data behind abstractions and expose functions that operate on that data. Data structures expose their data and have no meaningful functions.
*   **The Law of Demeter:** A module should not know about the innards of the objects it manipulates. Avoid "train wrecks" (e.g., `a.getB().getC().doSomething()`).

## Error Handling
*   **Prefer Exceptions:** Use exceptions rather than returning error codes.
*   **Isolate Error Handling:** Error handling is "one thing." Extract `try/catch` blocks into functions of their own to separate error handling from business logic.

## Tests and TDD
*   **The Three Laws of TDD:** Write a failing automated test before you write any code. You are not allowed to write more production code than is sufficient to pass the currently failing unit test.
*   **Clean Tests:** Tests must be maintained to the same high standard as production code. Use the Arrange-Act-Assert (Build-Operate-Check) pattern.
*   **F.I.R.S.T.:** Tests must be Fast, Independent, Repeatable, Self-Validating, and Timely.

## Classes
*   **Small & SRP:** Classes should be small, measured in responsibilities. A class should have a single reason to change (Single Responsibility Principle).
*   **Cohesion:** Classes should have a small number of instance variables, and each method should manipulate one or more of those variables.
