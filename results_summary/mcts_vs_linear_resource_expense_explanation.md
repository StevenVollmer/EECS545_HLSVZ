It's structural — the MCTS architecture creates a bimodal outcome by design, whereas linear agents have a continuous spectrum.

  Why MCTS peaks sharply at 5:                                                                                                                                                                                          
   
  The MCTS search uses the value function to rank candidate patches at each node. When a problem is solvable by the model, the planner's plan + the value function guidance tends to steer the coder toward the correct 
  patch very quickly — usually within the first 4–5 nodes. Once the reviewer accepts, the run stops immediately. So solvable problems cluster tightly because the search is guided: it's not wandering, it's exploiting
  a narrow promising path.                                                                                                                                                                                              
                                                            
  The result is essentially two modes:                                                                                                                                                                                  
  - The model "gets it" → solves in 4–6 nodes (round 1 succeeds fast)
  - The model doesn't "get it" → exhausts 19 nodes in round 1, reviewer rejects, exhausts 19 more in round 2 → 38 total                                                                                                 
                                                                                                                       
  There's almost no middle ground because the architecture doesn't allow partial progress to carry over between rounds. Each round resets to HEAD.                                                                      
                                                                                                                                                                                                                        
  Why linear is wider (4–15):                                                                                                                                                                                           
                                                                                                                                                                                                                        
  Linear agents have no value function guiding which direction to pursue, so they explore more freely. Easy problems get solved in a few turns, harder problems require more back-and-forth with the environment        
  (running tests, reading errors, iterating). This produces a smooth spread across the budget rather than a sharp peak.
                                                                                                                                                                                                                        
  In short: MCTS's tight peak is evidence the guided search is working — when it works, it works fast. The sharpness isn't luck, it's the value function doing its job.   