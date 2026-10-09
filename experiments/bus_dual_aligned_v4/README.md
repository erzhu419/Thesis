# Dual Control Alignment

This branch restores the successful global chronological bus formulation and
adds continuous speed control alongside exact zero-or-positive holding. It does
not replace any earlier actor, dataset, evaluation, or manuscript result.

The unsuccessful station branch changed more than action dimension: passenger
ledger intervals overlap across vehicles; rewards are centered; zero holding is
only approached by a tanh actor; and critic, backup, and encoder settings differ.
These differences are observed implementation facts, not individually proven
causes. Frozen-policy speed interventions separate the actuator effect from
those learning changes.

The new passenger return telescopes to system cost, including the terminal tail.
The existing rigid/ridge headway function is a separate selectable objective.
Both share SUMO offline data and source-simulator interaction, with no SUMO
online training. Paired methods reuse the same offline initializer. H2O+ retains
ratio correction; the matched backbone omits it; MC-WM adds semantic world
models and normalized Bellman trust with the established 0.75/0.25 TD mixture.

Direct and common guarded executors are evaluated separately. Previously used
test days are exploratory replication, not new untouched confirmation. All
prespecified seeds and endpoints are retained; superiority is not a success
condition for the runtime gate. Analysis stays on the server; only the compact
summary is retrieved locally.
