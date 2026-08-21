"""Core trading library.

Deliberately thin. Importing `core` used to pull in the profile table and a
Postgres engine as a side effect, which meant a script that only wanted the cost
model paid for a database connection. Import from the module you need.
"""
