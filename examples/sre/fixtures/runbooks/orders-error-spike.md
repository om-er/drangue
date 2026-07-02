# Orders: error spike after deploy (slow queries / missing index)

Symptom: orders 5xx rate climbs within an hour of a deploy; checkout latency
rises across web and api-gateway; database CPU on the orders store is elevated.

Known cause: a deploy that changes the order-lookup query path can drop the
covering index, turning point reads into scans.

Remediation (reversible): restart the orders service, which restores its
degraded state and rebuilds the index. Safe to repeat; low blast radius.

Escalate instead if: payments is also degraded (provider slowness — do NOT
restart payments; it makes the incident worse), or the spike predates the
deploy by more than an hour.
