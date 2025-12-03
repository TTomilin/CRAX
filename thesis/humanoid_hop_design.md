# Humanoid-Hop Environment Design Document

**Date**: October 27, 2025  
**Purpose**: Design a novel contact-based constraint environment for SafeBrax thesis expansion  
**Timeline**: Complete design, implementation, and evaluation within 3-4 days

---

## 1. Motivation & Research Goal

### Why One-Legged Hopping?
- **Tests contact-based constraints** (binary) vs existing continuous constraints (velocity, height)
- **Demonstrates rapid environment development** capability of SafeBrax
- **Real-world relevance**: Damage recovery, uneven terrain navigation, rehabilitation robotics
- **Technically interesting**: Requires dynamic balance, coordination, and precise timing

### Contribution to Thesis
- Proves SafeBrax enables custom constraint design in days (not weeks)
- Extends constraint taxonomy: adds contact pattern constraints
- Shows framework generality beyond velocity/height/collision constraints
- Provides evidence for "rapid experimentation" claim

---

## 2. Environment Specification

### Task Description
The humanoid must move forward while hopping on ONE designated leg. The opposite leg must remain off the ground (or minimally touching). This creates a challenging locomotion task that requires:
- Dynamic balance on a single leg
- Coordinated arm swinging for stability
- Timed push-off and landing
- Forward momentum while satisfying safety constraint

### Constraint Options (Pick One)

#### Option A: Strict Binary Contact Constraint
**Constraint**: Designated "hopping leg" can touch ground; opposite leg CANNOT touch ground  
**Cost**: Binary violation (1.0 if wrong foot touches, 0.0 otherwise)
```
c_t = 1.0 if contact_force_wrong_foot > threshold else 0.0
```
**Pros**: Clear, interpretable, matches real safety scenarios  
**Cons**: Sparse signal, may be hard to learn initially

#### Option B: Soft Contact Force Constraint  
**Constraint**: Limit contact forces on non-hopping leg  
**Cost**: Continuous penalty proportional to contact force
```
c_t = w_contact * max(0, contact_force_wrong_foot - f_threshold)
```
**Pros**: Dense gradient signal, easier to learn  
**Cons**: Less strict safety guarantee

#### Option C: Contact Time Budget Constraint
**Constraint**: Non-hopping foot can touch briefly but limited cumulative time  
**Cost**: Penalty when cumulative contact time exceeds budget
```
cumulative_contact_time += (contact_force_wrong_foot > 0) * dt
c_t = max(0, cumulative_contact_time - budget)
```
**Pros**: Allows realistic "stabilization touches", more forgiving  
**Cons**: More complex, requires state tracking

**DECISION**: Start with **Option A (Binary)** for clear constraint satisfaction metrics, can extend to B/C if needed

---

## 3. Technical Specification

### Observation Space
**Base**: Standard Brax Humanoid observations (376-dim)
- Torso position (z) and orientation (quaternion)
- Joint angles (17 actuated joints)
- Linear and angular velocities
- Joint angular velocities
- Physics quantities (inertia, forces)

**Added**: Contact sensors (2-4 additional dimensions)
- Left foot contact force (scalar)
- Right foot contact force (scalar)
- [Optional] Left foot contact binary flag
- [Optional] Right foot contact binary flag

**Total**: ~378-380 dimensions

### Action Space
**Same as standard humanoid**: 17-dimensional continuous
- 3 abdomen joints (yaw, pitch, roll)
- 6 hip joints (3 DOF per leg)
- 2 knee joints
- 4 shoulder joints (2 DOF per arm)
- 2 elbow joints

### Designated Hopping Leg
**Configuration options**:
1. **Fixed left leg**: Always hop on left, right must stay up
2. **Fixed right leg**: Always hop on right, left must stay up
3. **Randomized per episode**: 50% left, 50% right (tests generalization)
4. **User configurable**: Parameter in environment config

**DECISION**: Start with **Fixed left leg** for simplicity, add randomization as parameter later

---

## 4. Reward Function Design

### Primary Objective: Forward Progress
Base reward on forward (x-axis) velocity:
```
r_forward = v_x
```

### Additional Reward Components

#### Option 1: Simple Forward Velocity
```python
r_t = v_x + r_healthy - w_ctrl * ||a_t||^2
```
- `v_x`: Forward velocity (m/s)
- `r_healthy`: 1.0 per timestep upright
- `w_ctrl`: 0.01 control cost weight

#### Option 2: Add Balance Incentive
```python
r_t = v_x + r_healthy + w_balance * balance_bonus - w_ctrl * ||a_t||^2
```
where `balance_bonus = 1.0` if torso angle deviation < threshold

#### Option 3: Add Hop Quality Metrics
```python
r_t = v_x + r_healthy + w_height * hop_height - w_ctrl * ||a_t||^2
```
Rewards achieving vertical height during hop

**DECISION**: Start with **Option 1 (Simple)** to isolate constraint learning, can add shaping if learning fails

---

## 5. Cost Function Design

### Binary Contact Violation Cost

```python
def compute_cost(contact_forces, hopping_leg_id):
    """
    Args:
        contact_forces: Dict with 'left_foot' and 'right_foot' forces
        hopping_leg_id: 'left' or 'right'
    Returns:
        cost: 1.0 if violation, 0.0 otherwise
    """
    non_hopping_leg = 'right_foot' if hopping_leg_id == 'left' else 'left_foot'
    contact_threshold = 0.1  # Newtons, minimal force to count as contact
    
    violation = contact_forces[non_hopping_leg] > contact_threshold
    cost = 1.0 if violation else 0.0
    
    return cost
```

### Episodic Cost Accumulation
With episode length H=2000, safety bound d=0.05:
- Episode cost limit: C_limit = d * H = 100
- Interpretation: Non-hopping foot can touch at most 100 steps out of 2000
- Percentage: 5% violation tolerance

### Alternative Tighter Bounds
- d=0.025 → C_limit=50 (2.5% tolerance)
- d=0.01 → C_limit=20 (1% tolerance)
- d=0.001 → C_limit=2 (near-zero tolerance, very strict)

**DECISION**: Test with d ∈ {0.01, 0.025, 0.05} to show constraint range

---

## 6. Implementation Plan

### Step 1: MuJoCo XML Modification
**Base**: Use existing Brax humanoid XML (`brax/envs/assets/humanoid.xml`)

**Modifications needed**:
1. Add contact sensors to feet
   ```xml
   <sensor>
       <touch name="left_foot_contact" site="left_foot"/>
       <touch name="right_foot_contact" site="right_foot"/>
   </sensor>
   ```
2. Verify foot site definitions exist (should be in base humanoid)
3. No geometry changes needed

**Estimated time**: 1-2 hours

### Step 2: Environment Class Implementation
**Location**: `safe_brax/envs/humanoid_hop.py`

**Structure**:
```python
from brax.envs.humanoid import Humanoid
from brax import base
import jax.numpy as jnp

class HumanoidHop(Humanoid):
    """Humanoid with one-legged hopping constraint."""
    
    def __init__(
        self,
        hopping_leg: str = 'left',
        height_limit: float = 1.3,
        contact_threshold: float = 0.1,
        cost_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hopping_leg = hopping_leg
        self.contact_threshold = contact_threshold
        self.cost_weight = cost_weight
        
    def reset(self, rng):
        # Standard humanoid reset
        state = super().reset(rng)
        # Add contact sensor readings to state info
        return state
        
    def step(self, state, action):
        # Standard physics step
        next_state = super().step(state, action)
        
        # Extract contact forces
        contact_forces = self._get_contact_forces(next_state.pipeline_state)
        
        # Compute violation cost
        cost = self._compute_contact_cost(contact_forces)
        
        # Add cost to info dict
        next_state.info['cost'] = cost
        next_state.info['left_foot_contact'] = contact_forces['left']
        next_state.info['right_foot_contact'] = contact_forces['right']
        
        return next_state
        
    def _get_contact_forces(self, pipeline_state):
        # Extract from MJX contact data
        # Map contact sensor indices to left/right foot
        pass
        
    def _compute_contact_cost(self, contact_forces):
        # Binary violation check
        non_hopping_foot = 'right' if self.hopping_leg == 'left' else 'left'
        violation = contact_forces[non_hopping_foot] > self.contact_threshold
        return self.cost_weight * float(violation)
```

**Estimated time**: 4-6 hours (including debugging)

### Step 3: Testing & Validation
1. **Random policy test**: Verify cost computation, no crashes
2. **Visualization**: Render episodes, confirm contact detection accurate
3. **Cost statistics**: Run 100 random episodes, verify cost range reasonable
4. **Observation space**: Confirm dimensions correct, no NaN/Inf values

**Estimated time**: 2-3 hours

### Step 4: Training with PPO-Lagrangian
1. Reuse existing training script
2. Run with multiple safety bounds: d ∈ {0.01, 0.025, 0.05}
3. Run 20 seeds per configuration
4. Training time: ~2-4 hours per configuration on RTX 3060

**Estimated time**: 8-12 hours (mostly GPU time, can run overnight)

---

## 7. Expected Results & Analysis

### Hypotheses
1. **Constraint satisfaction**: PPO-Lagrangian should drive costs toward C_limit
2. **Performance tradeoff**: Tighter bounds → lower forward velocity
3. **Learning difficulty**: Contact constraints may be harder than continuous (sparse signal)
4. **Emergence of hopping gait**: Visual inspection should show actual hopping behavior

### Metrics to Report
- Final episodic reward (mean ± 95% CI)
- Final episodic cost (mean ± 95% CI)
- Constraint satisfaction rate (% episodes where cost < C_limit)
- Forward distance traveled
- Average hop frequency (if measurable from contact patterns)
- Training convergence time

### Comparison Points
- **Humanoid-Height**: Both humanoid, both continuous constraints → compare learning speed
- **Point-Goal**: Both binary constraints → compare constraint satisfaction rates
- **Ant-Velocity**: Compare forward locomotion performance

---

## 8. Thesis Integration Plan

### Chapter Structure: "Rapid Environment Development Case Study"

**Section 5.1: Motivation**
- Contact-based constraints in real robotics
- Limitations of existing SafeBrax constraint types
- Goal: Demonstrate framework extensibility
- Research question: Can SafeBrax enable rapid custom constraint development?

**Section 5.2: Design Process** (2-3 pages)
- Constraint selection rationale
- Reward/cost formulation considerations
- Implementation strategy
- Timeline breakdown with evidence

**Section 5.3: Humanoid-Hop Specification** (2-3 pages)
- Full environment description (mirror structure of Ch 3)
- Observation/action spaces
- Reward and cost functions
- Comparison with existing environments

**Section 5.4: Experimental Evaluation** (3-4 pages)
- Training methodology
- Learning curves (reward/cost)
- Constraint satisfaction analysis
- Comparison with continuous constraints
- Qualitative analysis (emergent hopping behavior)

**Section 5.5: Development Timeline Analysis** (1-2 pages)
- Time breakdown: design → implementation → training
- Bottlenecks and solutions
- Comparison: "Traditional benchmarks require weeks"
- Evidence of SafeBrax advantage

**Section 5.6: Discussion** (1-2 pages)
- Contact vs continuous constraints: lessons learned
- Implications for Safe RL research
- When to use custom environments
- Future directions (multi-leg patterns, adaptive constraints)

**Total estimated pages**: 10-14 pages

---

## 9. Risk Mitigation

### Risk 1: Contact Detection Issues
**Problem**: MJX contact sensors may not provide clean signals  
**Mitigation**: 
- Add filtering/smoothing
- Fallback to geometric collision detection
- Adjust contact threshold

### Risk 2: Task Too Hard / Impossible
**Problem**: Hopping on one leg may be dynamically infeasible  
**Mitigation**:
- Add reward shaping (hop height bonus)
- Relax constraint (allow brief touches with Option C)
- Reduce episode length
- Start with unconstrained PPO to verify feasibility

### Risk 3: Learning Doesn't Converge
**Problem**: PPO-Lagrangian fails to learn policy  
**Mitigation**:
- Curriculum learning (start with loose bound, tighten)
- Pretrain on Humanoid-Height, transfer to Hop
- Adjust Lagrangian learning rate
- Add more exploration (higher entropy coefficient)

### Risk 4: Time Overrun
**Problem**: Implementation takes longer than 3-4 days  
**Mitigation**:
- Simplify constraint (binary only, no variants)
- Skip extensive ablations (1-2 bounds only)
- Reduce seed count (10 instead of 20)
- Focus on single result for thesis proof-of-concept

---

## 10. Next Steps (Action Items)

- [x] Create design document
- [ ] Review existing Brax humanoid environment code
- [ ] Locate and examine humanoid.xml file
- [ ] Implement contact sensor reading in MJX
- [ ] Create HumanoidHop environment class
- [ ] Test with random policy
- [ ] Visualize contact forces
- [ ] Train with PPO-Lagrangian (d=0.05)
- [ ] Analyze initial results
- [ ] Run full experiment suite (3 bounds × 20 seeds)
- [ ] Create plots for thesis
- [ ] Write case study chapter
- [ ] Generate timeline evidence/logs

---

## 11. Timeline (Detailed)

### Day 1 (Today - Oct 27)
- [x] Design document creation (1 hour)
- [ ] Code review of Brax humanoid (2 hours)
- [ ] Contact sensor implementation (3 hours)
- [ ] Initial testing (2 hours)

### Day 2 (Oct 28)
- [ ] Environment class implementation (4 hours)
- [ ] Debugging and validation (3 hours)
- [ ] First training run (overnight, start before end of day)

### Day 3 (Oct 29)
- [ ] Analyze initial results (2 hours)
- [ ] Tune hyperparameters if needed (2 hours)
- [ ] Launch full experiment suite (4 hours setup, overnight runs)

### Day 4 (Oct 30)
- [ ] Process results (3 hours)
- [ ] Create plots and tables (3 hours)
- [ ] Draft case study chapter (4 hours)

### Day 5 (Oct 31)
- [ ] Revise and polish chapter (3 hours)
- [ ] Integrate into thesis (2 hours)
- [ ] Final review (2 hours)

**Buffer**: Days 6-7 for unexpected issues

---

## 12. Notes & Ideas

### Potential Extensions (If Time Permits)
1. **Humanoid-Crawl**: Combine height + contact pattern (must use hands)
2. **Humanoid-Gallop**: Alternating leg pairs (asymmetric gait)
3. **Humanoid-Skip**: Both feet but alternating lead leg
4. **Adaptive hopping leg**: Switch legs mid-episode based on cost budget

### Visualization Ideas
- Heatmap of contact forces over time
- Trajectory visualization showing hop pattern
- Comparison video: unconstrained vs constrained humanoid

### Additional Metrics
- Hop efficiency: distance per hop
- Energy consumption: control effort per meter traveled
- Stability: torso orientation variance

---

**Status**: Design phase complete, ready for implementation  
**Next**: Review Brax humanoid code structure



