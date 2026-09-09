# Distributed-systems domain structure

This directory reserves the domain contract only. The first implementation
should cover request, downstream call, response or timeout, retry, terminal
outcome, and recovery. Parentage should come from trace context or an explicit
synthetic rule; topology alone must never create an event parent relation.
