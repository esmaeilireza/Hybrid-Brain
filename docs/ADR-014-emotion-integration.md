
## Update (post-implementation)
- XPASS confirmed and xfail marker REMOVED - ADR-005 exit condition met.
- The old test_unrewarded_episodes_not_replayed asserted valence-only
  replay (reward=-0.01 treated as unrewarded). That assertion encoded
  exactly the optimistic bias this ADR removes; it was replaced by two
  contracts: zero-reward guard + damped negative replay (|-0.002| over
  20 episodes vs online-scale). Design overturn, documented - not a
  test deleted to make a suite green.
